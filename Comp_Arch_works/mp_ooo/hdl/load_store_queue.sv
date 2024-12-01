module load_store_queue
import rv32im_types::*;
#(
    parameter LS_QUEUE_DEPTH = 8,
    parameter ROB_DEPTH = 8
) (
    input   logic                           clk,
    input   logic                           rst,
    input   logic                           flush,

    // decoder side signal
    input   logic                           iq_issue,
    input   logic [31:0]                    decoder_instr,
    input   logic [$clog2(ROB_DEPTH)-1:0]   decoder_tag,
    input   logic                           decoder_load_store_valid,
    output  logic                           lsq_full,


    // load_store_adder side signal
    input   logic                           addr_valid, //connect to lsq_addr_valid
    input   logic [31:0]                    load_store_addr, //connect to lsq_addr

    // RS side signal
    input   logic [31:0]                    ls_rs_store_data,  //connect to lsq_store_data
    input   logic [$clog2(ROB_DEPTH)-1:0]   ls_rs_tag, //connect to lsq_tag

    // dram port
    output  logic [31:0]                    d_dram_addr,
    output  logic                           d_dram_read,
    output  logic                           d_dram_write,
    output  logic [63:0]                    d_dram_wdata,

    input   logic                           d_dram_ready,
    input   logic [31:0]                    d_dram_raddr,
    input   logic [63:0]                    d_dram_rdata,
    input   logic                           d_dram_rvalid,

    // CDB side signal
    output  logic                           valid_CDB,
    output  logic [31:0]                    data_CDB,
    output  logic [$clog2(ROB_DEPTH)-1:0]   tag_CDB,

    // ROB side signal
    input   logic [$clog2(ROB_DEPTH)-1:0]   rob_commit_tag,

    output  logic [31:0]                    rob_dmem_addr, //for rvfi
    output  logic [3:0]                     rob_dmem_rmask, //for rvfi
    output  logic [3:0]                     rob_dmem_wmask, //for rvfi
    output  logic [31:0]                    rob_dmem_rdata, //for rvfi
    output  logic [31:0]                    rob_dmem_wdata //for rvfi

);
    logic [31:0]                         instr_arr[LS_QUEUE_DEPTH];
    logic                                valid_arr[LS_QUEUE_DEPTH]; // indicate this arr is occupied
    logic [31:0]                         addr_arr[LS_QUEUE_DEPTH];
    logic [31:0]                         dcache_wdata_arr[LS_QUEUE_DEPTH]; // data to write into dcache
    logic [31:0]                         dcache_rdata_buffer; // store data from dcache, next cycle give to rob
    logic [$clog2(ROB_DEPTH)-1:0]        tag_arr[LS_QUEUE_DEPTH]; 
    logic                                addr_ready_arr[LS_QUEUE_DEPTH];

    logic [$clog2(LS_QUEUE_DEPTH) - 1:0] lsq_head, lsq_tail;
    logic                                lsq_commit;

    logic [6:0]                          opcode;
    logic [2:0]                          funct3;

    // dcache side signal
    logic [31:0]                         dcache_addr;
    logic [3:0]                          dcache_rmask;
    logic [3:0]                          dcache_wmask;
    logic [31:0]                         dcache_wdata;
    logic [31:0]                         dcache_rdata;
    logic                                dcache_resp;
    // d_adapter side signal
    logic [31:0]                         d_adapter_addr;
    logic                                d_adapter_read;
    logic                                d_adapter_write;
    logic [255:0]                        d_adapter_rdata;
    logic [255:0]                        d_adapter_wdata;
    logic                                d_adapter_resp;

    assign opcode = instr_arr[lsq_head][6:0];
    assign funct3 = instr_arr[lsq_head][14:12];

    //lsq_full logic
    always_comb begin
        lsq_full  = 1'b1;  
        for (int i = 0; i < LS_QUEUE_DEPTH; i++) begin
            lsq_full  &= valid_arr[i];
        end 
    end
    
    //commit logic
    always_ff @(posedge clk) begin
        lsq_commit <= 1'b0;
        if (valid_arr[lsq_head] && dcache_resp) begin
            lsq_commit <= 1'b1;
        end
    end

    //head tail logic
    always_ff @(posedge clk) begin
        if (rst || flush) begin
            lsq_head <= '0;
            lsq_tail <= '0;
        end
        else begin
            if (iq_issue && decoder_load_store_valid) begin
                lsq_tail <= lsq_tail + ($clog2(LS_QUEUE_DEPTH))'(1);
            end
            if (lsq_commit) begin
                lsq_head <= lsq_head + ($clog2(LS_QUEUE_DEPTH))'(1);
            end
        end
    end

    //lsq data logic
    always_ff @(posedge clk) begin
        if (rst | flush) begin
            for (int i = 0; i < LS_QUEUE_DEPTH; i++) begin
                valid_arr[i]          <= '0;
                instr_arr[i]          <= '0;
                addr_arr[i]           <= '0;
                dcache_wdata_arr[i]   <= '0;
                tag_arr[i]            <= '0;
                addr_ready_arr[i]     <= '0;
            end
        end
        else begin
            if (lsq_commit) begin
                valid_arr[lsq_head]           <= '0;
                instr_arr[lsq_head]           <= '0;
                addr_arr[lsq_head]            <= '0;
                dcache_wdata_arr[lsq_head]    <= '0;
                tag_arr[lsq_head]             <= '0;
                addr_ready_arr[lsq_head]      <= '0;
            end
            if (iq_issue && decoder_load_store_valid) begin
                valid_arr[lsq_tail]   <= decoder_load_store_valid;
                instr_arr[lsq_tail]   <= decoder_instr;
                tag_arr[lsq_tail]     <= decoder_tag;
            end
            if (addr_valid) begin
                for (int unsigned i = 0; i < LS_QUEUE_DEPTH; i++) begin
                    if(valid_arr[i] && tag_arr[i] == ls_rs_tag) begin
                        addr_arr[i]         <= load_store_addr;
                        dcache_wdata_arr[i]   <= ls_rs_store_data;
                        addr_ready_arr[i]     <= addr_valid;
                    end
                end
            end
        end
    end
    
    //dcache FSM
    enum logic [1:0] {
        IDLE,
        LOAD,
        STORE,
        DONE
    } state, state_next;

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
        end
        else begin
            state <= state_next;
        end
    end

    //dcache FSM
    always_comb begin
        dcache_addr  = '0;
        dcache_rmask = '0;
        dcache_wmask = '0;
        dcache_wdata = '0;
        state_next = IDLE;
        case(state)
            IDLE: begin
                if (addr_ready_arr[lsq_head] && valid_arr[lsq_head] && (rob_commit_tag == tag_arr[lsq_head])) begin
                    if (opcode == op_b_load) begin
                        state_next = LOAD;
                        case(funct3)
                            load_f3_lb: begin
                                dcache_rmask = 4'b0001 << addr_arr[lsq_head][1:0];
                                dcache_addr = addr_arr[lsq_head];
                            end
                            load_f3_lbu: begin
                                dcache_rmask = 4'b0001 << addr_arr[lsq_head][1:0];
                                dcache_addr = addr_arr[lsq_head];
                            end
                            load_f3_lh: begin
                                dcache_rmask = 4'b0011 << addr_arr[lsq_head][1:0];
                                dcache_addr = addr_arr[lsq_head];
                            end
                            load_f3_lhu: begin
                                dcache_rmask = 4'b0011 << addr_arr[lsq_head][1:0];
                                dcache_addr = addr_arr[lsq_head];
                            end
                            load_f3_lw: begin
                                dcache_rmask = 4'b1111;
                                dcache_addr = addr_arr[lsq_head];
                            end
                            default: begin
                                dcache_rmask = 4'b0000;
                                dcache_addr = '0;
                            end
                        endcase
                    end
                    else if (opcode == op_b_store) begin
                        state_next = STORE;
                        case (funct3)
                            store_f3_sb: begin
                                dcache_addr = addr_arr[lsq_head];
                                dcache_wmask = 4'b0001 << addr_arr[lsq_head][1:0];
                                dcache_wdata[8 *addr_arr[lsq_head][1:0] +: 8 ] = dcache_wdata_arr[lsq_head][7:0];
                            end
                            store_f3_sh: begin
                                dcache_addr = addr_arr[lsq_head];
                                dcache_wmask = 4'b0011 << addr_arr[lsq_head][1:0];
                                dcache_wdata[16 *addr_arr[lsq_head][1] +: 16 ] = dcache_wdata_arr[lsq_head][15:0];
                            end
                            store_f3_sw: begin
                                dcache_addr = addr_arr[lsq_head];
                                dcache_wmask = 4'b1111;
                                dcache_wdata = dcache_wdata_arr[lsq_head];
                            end
                            default: begin
                                dcache_addr = '0;
                                dcache_wmask = 4'b0000;
                                dcache_wdata = '0;
                            end
                        endcase
                    end
                    else begin
                        state_next = IDLE;
                        dcache_addr  = '0;
                        dcache_rmask = '0;
                        dcache_wmask = '0;
                        dcache_wdata = '0;
                    end
                end
            end
            LOAD: begin
                if (dcache_resp) begin
                    state_next = DONE;
                    dcache_rmask = '0;
                    dcache_wmask = '0;
                    dcache_wdata = '0;
                end
                else begin
                    state_next = LOAD;
                    dcache_rmask = 'x;
                    dcache_wmask = 'x;
                    dcache_wdata = 'x;
                end

            end
            STORE: begin
                if (dcache_resp) begin
                    state_next = DONE;
                    dcache_rmask = '0;
                    dcache_wmask = '0;
                    dcache_wdata = '0;
                end
                else begin
                    state_next = STORE;
                    dcache_rmask = 'x;
                    dcache_wmask = 'x;
                    dcache_wdata = 'x;
                end

            end
            DONE: begin
                state_next = IDLE;
                dcache_rmask = '0;
                dcache_wmask = '0;
                dcache_wdata = '0;
            end
            default: begin
                state_next = IDLE;
                dcache_rmask = '0;
                dcache_wmask = '0;
                dcache_wdata = '0;
            end
        endcase
    end
    
    // dcache_rdata_buffer logic
    always_ff @(posedge clk) begin
        if (rst) begin
            dcache_rdata_buffer <= '0;
        end
        else begin
            case(state) 
                IDLE: begin
                    dcache_rdata_buffer <= '0;
                end
                LOAD: begin
                    if (dcache_resp) begin
                        case (funct3)
                            load_f3_lb: begin
                                dcache_rdata_buffer <= {{24{dcache_rdata[7 +8 *addr_arr[lsq_head][1:0]]}}, dcache_rdata[8 *addr_arr[lsq_head][1:0] +: 8 ]};
                            end
                            load_f3_lbu: begin
                                dcache_rdata_buffer <= {{24{1'b0}}, dcache_rdata[8 *addr_arr[lsq_head][1:0] +: 8 ]};
                            end
                            load_f3_lh: begin
                                dcache_rdata_buffer <= {{16{dcache_rdata[15 +16 *addr_arr[lsq_head][1]]}}, dcache_rdata[16 *addr_arr[lsq_head][1] +: 16 ]};
                            end
                            load_f3_lhu: begin
                                dcache_rdata_buffer <= {{16{1'b0}}, dcache_rdata[16 *addr_arr[lsq_head][1] +: 16 ]};
                            end
                            load_f3_lw: begin
                                dcache_rdata_buffer <= dcache_rdata;
                            end
                            default: begin
                                dcache_rdata_buffer <= dcache_rdata;
                            end
                        endcase
                    end
                end
                STORE: begin
                    dcache_rdata_buffer <= '0;
                end
                DONE: begin
                    dcache_rdata_buffer <= '0;
                end
                default: begin
                    dcache_rdata_buffer <= '0;
                end
            endcase
        end
    end
    
    // rvfi signal logic
    always_ff @(posedge clk) begin
        if (rst) begin
            rob_dmem_addr  <= '0;
            rob_dmem_rmask <= '0;
            rob_dmem_wmask <= '0;
            rob_dmem_rdata <= '0;
            rob_dmem_wdata <= '0;
        end
        else begin
            case(state)
                IDLE: begin
                    if (valid_arr[lsq_head] && rob_commit_tag == tag_arr[lsq_head]) begin
                        rob_dmem_addr <= addr_arr[lsq_head];
                        if (opcode == op_b_load) begin
                            rob_dmem_rmask <= dcache_rmask;
                        end
                        if (opcode == op_b_store) begin
                            rob_dmem_wmask <= dcache_wmask;
                            rob_dmem_wdata <= dcache_wdata;     
                        end
                    end
                    else begin
                        rob_dmem_addr  <= '0;
                        rob_dmem_rmask <= '0;
                        rob_dmem_wmask <= '0;
                        rob_dmem_rdata <= '0;
                        rob_dmem_wdata <= '0;
                    end
                end
                LOAD: begin
                    if (dcache_resp) begin
                        rob_dmem_rdata <= dcache_rdata;
                    end
                end
                STORE: begin
                    rob_dmem_addr  <= rob_dmem_addr;
                    rob_dmem_rmask <= rob_dmem_rmask;
                    rob_dmem_wmask <= rob_dmem_wmask;
                    rob_dmem_rdata <= rob_dmem_rdata;
                    rob_dmem_wdata <= rob_dmem_wdata;
                
                end
                DONE: begin
                    rob_dmem_addr  <= '0;
                    rob_dmem_rmask <= '0;
                    rob_dmem_wmask <= '0;
                    rob_dmem_rdata <= '0;
                    rob_dmem_wdata <= '0;
                end
                default: begin
                    rob_dmem_addr  <= '0;
                    rob_dmem_rmask <= '0;
                    rob_dmem_wmask <= '0;
                    rob_dmem_rdata <= '0;
                    rob_dmem_wdata <= '0;
                end
            endcase
        end
    end

    // CDB logic
    always_comb begin
        valid_CDB = 1'b0;
        data_CDB = '0;
        tag_CDB = '0;
        case(state)
            IDLE: begin
                valid_CDB = 1'b0;
                data_CDB = '0;
                tag_CDB = '0;
            end
            LOAD: begin
                valid_CDB = 1'b0;
                data_CDB = '0;
                tag_CDB = '0;
                // if (dcache_resp) begin
                //     valid_CDB = 1'b1;
                //     tag_CDB = tag_arr[lsq_head];
                //     data_CDB = dcache_rdata;
                // end
            end
            STORE: begin
                valid_CDB = 1'b0;
                data_CDB = '0;
                tag_CDB = '0;
                if (dcache_resp) begin
                    valid_CDB = 1'b1;
                    tag_CDB = tag_arr[lsq_head];
                end
            end
            DONE: begin
                valid_CDB = 1'b1;
                data_CDB = dcache_rdata_buffer;
                tag_CDB = tag_arr[lsq_head];
            end
            default: begin
                valid_CDB = 1'b0;
                data_CDB = '0;
                tag_CDB = '0;
            end
        endcase
    end


    provided_cache dcache (
        .clk        (clk),
        .rst        (rst),
        .ufp_addr   (dcache_addr),
        .ufp_rmask  (dcache_rmask),
        .ufp_wmask  (dcache_wmask),
        .ufp_rdata  (dcache_rdata),
        .ufp_wdata  (dcache_wdata),
        .ufp_resp   (dcache_resp),
        .dfp_addr   (d_adapter_addr),
        .dfp_read   (d_adapter_read),
        .dfp_write  (d_adapter_write),
        .dfp_rdata  (d_adapter_rdata),
        .dfp_wdata  (d_adapter_wdata),
        .dfp_resp   (d_adapter_resp)
    );

    cacheline_adapter dcache_adapter (
        .clk        (clk),
        .rst        (rst),
        .ufp_addr   (d_adapter_addr),
        .ufp_read   (d_adapter_read),
        .ufp_write  (d_adapter_write),
        .ufp_wdata  (d_adapter_wdata),
        .ufp_rdata  (d_adapter_rdata),
        .ufp_resp   (d_adapter_resp),
        .dfp_addr   (d_dram_addr),
        .dfp_read   (d_dram_read),
        .dfp_write  (d_dram_write),
        .dfp_wdata  (d_dram_wdata),
        .dfp_ready  (d_dram_ready),
        .dfp_raddr  (d_dram_raddr),
        .dfp_rdata  (d_dram_rdata),
        .dfp_rvalid (d_dram_rvalid) 
    );

endmodule
