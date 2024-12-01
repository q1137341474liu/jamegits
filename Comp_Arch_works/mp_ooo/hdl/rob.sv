module rob
import rv32im_types::*;
#(
    parameter ROB_DEPTH = 8,
    parameter CDB_SIZE = 4 //alu: 0, mul_div: 1, load_store: 2, branch: 3
) (
    // order + 1 when commit
    input   logic                           clk,
    input   logic                           rst,
    output   logic                          flush,

    // CDB side signal
    input   logic                           valid_CDB[CDB_SIZE],
    input   logic [$clog2(ROB_DEPTH) - 1:0] tag_CDB[CDB_SIZE],
    input   logic [31:0]                    data_CDB[CDB_SIZE],

    output  logic                           rob_commit,

    // decoder side signal
    input   logic                           iq_issue, 
    input   logic                           decoder_valid,
    input   logic [31:0]                    decoder_instr,
    input   logic [4:0]                     decoder_rs1_s,
    input   logic [4:0]                     decoder_rs2_s,
    input   logic [4:0]                     decoder_rd_s,
    input   logic                           decoder_regf_we,
    input   logic [31:0]                    decoder_pc,
    input   logic [31:0]                    decoder_pc_next,
    input   logic                           decoder_pc_next_valid,
    output  logic                           rob_full,
    
    // regfile side signal
    output  logic                           regf_commit_regf_we,
    output  logic [4:0]                     regf_commit_rd_s,
    output  logic [31:0]                    regf_commit_rd_v,
    output  logic [$clog2(ROB_DEPTH)-1:0]   regf_commit_tag,

    output  logic [4:0]                     regf_issue_rd_s,
    output  logic [$clog2(ROB_DEPTH)-1:0]   regf_issue_tag,

    // rs side signal
    input   logic [$clog2(ROB_DEPTH)-1:0]   rs_tag[CDB_SIZE], // connect to RS tag_dest_out output
    input   logic [31:0]                    rs_rs1_v[CDB_SIZE], // connect to RS data_A_out output
    input   logic [31:0]                    rs_rs2_v[CDB_SIZE], // connect to RS data_B_out output
    input   logic                           rs_ready[CDB_SIZE], // connect to RS comp_issue output


    // branch side signal
    input   logic [31:0]                    branch_pc_next,

    // fetch side signal
    output  logic [31:0]                    fetch_pc_next,

    // lsq side signal
    input   logic [31:0]                    lsq_dmem_addr, //for rvfi
    input   logic [3:0]                     lsq_dmem_rmask, //for rvfi
    input   logic [3:0]                     lsq_dmem_wmask, //for rvfi
    input   logic [31:0]                    lsq_dmem_rdata, //for rvfi
    input   logic [31:0]                    lsq_dmem_wdata, //for rvfi
    output  logic [$clog2(ROB_DEPTH)-1:0]   lsq_commit_tag
    // rvfi connection signal
    //output  rvfi_t                          rvfi
);
    rvfi_t                          rvfi;

    logic                           valid_arr[ROB_DEPTH];
    logic [31:0]                    instr_arr[ROB_DEPTH];
    logic [4:0]                     rs1_s_arr[ROB_DEPTH];
    logic [4:0]                     rs2_s_arr[ROB_DEPTH];
    logic [31:0]                    rs1_v_arr[ROB_DEPTH];
    logic [31:0]                    rs2_v_arr[ROB_DEPTH];
    logic [4:0]                     rd_s_arr[ROB_DEPTH];
    logic [31:0]                    rd_v_arr[ROB_DEPTH];
    logic [31:0]                    pc_arr[ROB_DEPTH];
    logic [31:0]                    pc_next_arr[ROB_DEPTH];
    logic                           pc_next_valid_arr[ROB_DEPTH];
    logic [31:0]                    dmem_addr;
    logic [3:0]                     dmem_rmask;
    logic [3:0]                     dmem_wmask;
    logic [31:0]                    dmem_rdata;
    logic [31:0]                    dmem_wdata;

   
    logic                           regf_we_arr[ROB_DEPTH];
    logic                           commit_ready_arr [ROB_DEPTH]; // verify rd_data is ready
    logic [63:0]                    order, order_next; //counter for commited instruction
    logic [$clog2(ROB_DEPTH) - 1:0] rob_head, rob_tail; // head pointing to commit row, tail pointing to issue row
   
    assign regf_commit_regf_we    = regf_we_arr[rob_head];
    assign regf_commit_rd_s       = rd_s_arr[rob_head];
    assign regf_commit_rd_v       = rd_v_arr[rob_head];
    assign regf_commit_tag        = rob_head;
    assign regf_issue_rd_s        = decoder_rd_s;
    assign regf_issue_tag         = rob_tail;
    assign lsq_commit_tag         = rob_head;

    //rob_full logic
    always_comb begin
        rob_full  = 1'b1;  
        for (int i = 0; i < ROB_DEPTH; i++) begin
            rob_full  &= valid_arr[i];
        end 
    end
    
    //commit logic
    always_comb begin
        rob_commit = 1'b0;
        if (valid_arr[rob_head] & commit_ready_arr[rob_head]) begin
            rob_commit = 1'b1;
        end
    end

    //flush logic
    always_comb begin
        flush = '0;
        fetch_pc_next = 'x;
        if(rob_commit) begin
            if(pc_next_arr[rob_head] == pc_arr[rob_head] +'d4 ) begin
                flush = '0;
            end
            else begin
                flush = 1'b1;
                fetch_pc_next = pc_next_arr[rob_head];
            end
        end
    end

    //head tail logic
    always_ff @(posedge clk) begin
        if (rst || flush) begin
            rob_head <= '0;
            rob_tail <= '0;
        end
        else begin
            if (iq_issue) begin
                rob_tail <= rob_tail + ($clog2(ROB_DEPTH))'(1);
            end
            if (rob_commit) begin
                rob_head <= rob_head + ($clog2(ROB_DEPTH))'(1);
            end
        end
    end

    //rob data logic
    always_ff @(posedge clk) begin
        if (rst) begin
            order       <= '0;
        end
        if (rst | flush) begin
            if (rob_commit) begin
                order <= order + 'd1;
            end
            for (int i = 0; i < ROB_DEPTH; i++) begin
                valid_arr[i]            <= '0;
                instr_arr[i]            <= '0;
                rs1_s_arr[i]            <= '0;
                rs2_s_arr[i]            <= '0;
                rs1_v_arr[i]            <= '0;
                rs2_v_arr[i]            <= '0;
                rd_s_arr[i]             <= '0;
                rd_v_arr[i]             <= '0;
                pc_arr[i]               <= '0;
                pc_next_arr[i]          <= '0;
                pc_next_valid_arr[i]    <= '0;
                regf_we_arr[i]          <= '0;
                commit_ready_arr [i]    <= '0;
            end
            dmem_addr   <= '0;
            dmem_rmask  <= '0;
            dmem_wmask  <= '0;
            dmem_rdata  <= '0;
            dmem_wdata  <= '0;
        end
        else begin
            for (int unsigned i = 0; i < ROB_DEPTH; i++) begin
                for (int j = 0; j < CDB_SIZE; j++) begin
                    if (!commit_ready_arr[i] && valid_arr[i] && valid_CDB[j]) begin 
                        if (tag_CDB[j] == ($clog2(ROB_DEPTH))'(i)) begin
                            dmem_addr             <= lsq_dmem_addr;
                            dmem_rmask            <= lsq_dmem_rmask;
                            dmem_wmask            <= lsq_dmem_wmask;
                            dmem_rdata            <= lsq_dmem_rdata;
                            dmem_wdata            <= lsq_dmem_wdata;
                            if (regf_we_arr[i]) begin
                                rd_v_arr[i]           <= data_CDB[j];
                                commit_ready_arr[i]   <= 1'b1;
                                if (pc_next_valid_arr[i] == 1'b0) begin
                                    pc_next_arr[i]        <= branch_pc_next;
                                end
                            end
                            if (!pc_next_valid_arr[i]) begin
                                pc_next_arr[i]        <= branch_pc_next;
                                commit_ready_arr[i]   <= 1'b1;
                            end
                            else begin // for store instruction
                                commit_ready_arr[i]   <= 1'b1;
                            end

                        end
                    end
                end
                for (int j = 0; j < CDB_SIZE; j++) begin
                    if (rs_ready[j] && valid_arr[i] && (rs_tag[j] == $clog2(ROB_DEPTH)'(i))) begin
                        rs1_v_arr[i] <= rs_rs1_v[j];
                        rs2_v_arr[i] <= rs_rs2_v[j];
                    end
                end
            end
            if (rob_commit) begin
                order                         <= order + 'd1;
                valid_arr[rob_head]           <= '0;
                instr_arr[rob_head]           <= '0;
                rs1_s_arr[rob_head]           <= '0;
                rs2_s_arr[rob_head]           <= '0;
                rs1_v_arr[rob_head]           <= '0;
                rs2_v_arr[rob_head]           <= '0;
                rd_s_arr[rob_head]            <= '0;
                rd_v_arr[rob_head]            <= '0;
                pc_arr[rob_head]              <= '0;
                pc_next_arr[rob_head]         <= '0;
                pc_next_valid_arr[rob_head]   <= '0;
                commit_ready_arr [rob_head]   <= '0;
                dmem_addr                     <= '0;
                dmem_rmask                    <= '0;
                dmem_wmask                    <= '0;
                dmem_rdata                    <= '0;
                dmem_wdata                    <= '0;
            end
            if (iq_issue) begin
                valid_arr[rob_tail]           <= decoder_valid;
                instr_arr[rob_tail]           <= decoder_instr;
                rs1_s_arr[rob_tail]           <= decoder_rs1_s;
                rs2_s_arr[rob_tail]           <= decoder_rs2_s;
                rd_s_arr[rob_tail]            <= decoder_rd_s;
                pc_arr[rob_tail]              <= decoder_pc;
                pc_next_arr[rob_tail]         <= decoder_pc_next;
                pc_next_valid_arr[rob_tail]   <= decoder_pc_next_valid;
                regf_we_arr[rob_tail]         <= decoder_regf_we;

            end
        end
    end

    
    //assign rvfi signal
    always_comb begin
        rvfi.valid      = rob_commit;
        rvfi.order      = order;
        rvfi.inst       = instr_arr[rob_head];
        rvfi.rs1_addr   = rs1_s_arr[rob_head];
        rvfi.rs2_addr   = rs2_s_arr[rob_head];
        rvfi.rs1_rdata  = rs1_v_arr[rob_head];
        rvfi.rs2_rdata  = rs2_v_arr[rob_head];
        rvfi.rd_addr    = rd_s_arr[rob_head];
        rvfi.rd_wdata   = rd_v_arr[rob_head];
        rvfi.pc_rdata   = pc_arr[rob_head];
        rvfi.pc_wdata   = pc_next_arr[rob_head];
        rvfi.dmem_addr  = dmem_addr;
        rvfi.dmem_rmask = dmem_rmask;
        rvfi.dmem_wmask = dmem_wmask;
        rvfi.dmem_rdata = dmem_rdata;
        rvfi.dmem_wdata = dmem_wdata;
    end

endmodule
