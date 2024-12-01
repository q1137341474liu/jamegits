
module cpu
import rv32i_types::*;
(
    input   logic           clk,
    input   logic           rst,

    output  logic   [31:0]  imem_addr,
    output  logic   [3:0]   imem_rmask,
    input   logic   [31:0]  imem_rdata,
    input   logic           imem_resp,

    output  logic   [31:0]  dmem_addr,
    output  logic   [3:0]   dmem_rmask,
    output  logic   [3:0]   dmem_wmask,
    input   logic   [31:0]  dmem_rdata,
    output  logic   [31:0]  dmem_wdata,
    input   logic           dmem_resp
);

    if_id_stage_reg_t   curr_if_id_stage_reg, next_if_id_stage_reg;
    id_ex_stage_reg_t   curr_id_ex_stage_reg, next_id_ex_stage_reg;
    ex_mem_stage_reg_t  curr_ex_mem_stage_reg, next_ex_mem_stage_reg;
    mem_wb_stage_reg_t  curr_mem_wb_stage_reg, next_mem_wb_stage_reg;
    logic               stall;
    logic [4:0]         rd_s_wb;
    logic [31:0]        rd_v_wb;
    logic               regf_we_wb;
    rs1_forward_id_t         rs1_forward_id;
    rs2_forward_id_t         rs2_forward_id;
    rs1_forward_ex_t         rs1_forward_ex;
    rs2_forward_ex_t         rs2_forward_ex;
    logic  [4:0]    rs1_s_d;
    logic  [4:0]    rs2_s_d;
    logic  flush;
    logic  [31:0]  target_pc;
    logic branch_flush_delay;
    logic imem_rqst;
    logic dmem_rqst;
    logic go;

    stall stalllogic(
        .clk(clk),
        .rst(rst),
        .imem_need(imem_rqst),
        .dmem_need(dmem_rqst),
        .imem_resp(imem_resp),
        .dmem_resp(dmem_resp),
        .go(go)
    ); 

    Forwarding forwarding(
        .id_rs1_s(rs1_s_d),
        .id_rs2_s(rs2_s_d),
        .id_ex_stage_reg(curr_id_ex_stage_reg),
        .ex_mem_stage_reg(curr_ex_mem_stage_reg),
        .mem_wb_stage_reg(curr_mem_wb_stage_reg),
        .forwarding_stall(stall),
        .rs1_forward_id(rs1_forward_id),
        .rs2_forward_id(rs2_forward_id),
        .rs1_forward_ex(rs1_forward_ex),
        .rs2_forward_ex(rs2_forward_ex)
    );


    fetch fetch_stage(
        .clk(clk),
        .rst(rst),
        .imem_addr(imem_addr),
        .imem_rmask(imem_rmask),
        .stall(stall),
        .if_id_stage_reg(next_if_id_stage_reg),
        .flush(flush),
        .pc_result(target_pc),
        .go(go),
        .imem_trigger(imem_rqst)
    );

    always_ff @ (posedge clk ) begin 
        if (rst) begin 
            curr_if_id_stage_reg <= '0;
        end else if (~stall) begin 
            if (go) begin 
                curr_if_id_stage_reg <= next_if_id_stage_reg;
            end 
        end 
    end 

    decode decode_stage(
        .clk(clk),
        .rst(rst),
        .if_id_stage_reg(curr_if_id_stage_reg),
        .id_ex_stage_reg(next_id_ex_stage_reg),
        .imem_resp(imem_resp),
        .imem_rdata(imem_rdata),
        .rd_s_wb(rd_s_wb),
        .rd_v_wb(rd_v_wb),
        .regf_we_wb(regf_we_wb),
        .stall(stall),
        .rs1_s_d(rs1_s_d),
        .rs2_s_d(rs2_s_d),
        .rs1_forward_id(rs1_forward_id),
        .rs2_forward_id(rs2_forward_id),
        .flush(flush),
        .go(go)
    );

    always_ff @ (posedge clk ) begin 
        if (rst) begin 
            curr_id_ex_stage_reg <= '0;
        end else if (go) begin 
            curr_id_ex_stage_reg <= next_id_ex_stage_reg;
        end 
    end

    excute excute_stage(
        .id_ex_stage_reg(curr_id_ex_stage_reg),
        .ex_mem_stage_reg_out(next_ex_mem_stage_reg),
        .ex_mem_stage_reg_in(curr_ex_mem_stage_reg),
        .rd_v_wb(rd_v_wb),
        .rs1_forward_ex(rs1_forward_ex),
        .rs2_forward_ex(rs2_forward_ex),
        .flush(flush),
        .BrPC(target_pc)
    );

    always_ff @ (posedge clk ) begin 
        if (rst) begin 
            curr_ex_mem_stage_reg <= '0;
        end else if (go) begin 
            curr_ex_mem_stage_reg <= next_ex_mem_stage_reg;
        end 
    end

    memory mem_stage(
        .clk(clk),
        .rst(rst),
	.dmem_rdata(dmem_rdata),
        .dmem_addr(dmem_addr),
        .dmem_rmask(dmem_rmask),
        .dmem_wmask(dmem_wmask),
        .dmem_wdata(dmem_wdata),
	.dmem_resp(dmem_resp),
        .ex_mem_stage_reg(curr_ex_mem_stage_reg),
        .mem_wb_stage_reg(next_mem_wb_stage_reg),
        .dmem_need(dmem_rqst),
        .go(go)
    );

    always_ff @ (posedge clk ) begin 
        if (rst) begin 
            curr_mem_wb_stage_reg <= '0;
        end else if (go) begin 
            curr_mem_wb_stage_reg <= next_mem_wb_stage_reg;
        end 
    end

    writeback writeback_stage(
 
        .mem_wb_stage_reg(curr_mem_wb_stage_reg),
        .rd_s_wb(rd_s_wb),
        .rd_v_wb(rd_v_wb),
        .regf_we_wb(regf_we_wb),
        .go(go)
    );
/*
            logic           monitor_valid;
            logic   [63:0]  monitor_order;
            logic   [31:0]  monitor_inst;
            logic   [4:0]   monitor_rs1_addr;
            logic   [4:0]   monitor_rs2_addr;
            logic   [31:0]  monitor_rs1_rdata;
            logic   [31:0]  monitor_rs2_rdata;
            logic           monitor_regf_we;
            logic   [4:0]   monitor_rd_addr;
            logic   [31:0]  monitor_rd_wdata;
            logic   [31:0]  monitor_pc_rdata;
            logic   [31:0]  monitor_pc_wdata;
            logic   [31:0]  monitor_mem_addr;
            logic   [3:0]   monitor_mem_rmask;
            logic   [3:0]   monitor_mem_wmask;
            logic   [31:0]  monitor_mem_rdata;
            logic   [31:0]  monitor_mem_wdata;
    assign monitor_valid     = writeback_stage.valid;
    assign monitor_order     = writeback_stage.order;
    assign monitor_inst      = writeback_stage.inst;
    assign monitor_rs1_addr  = writeback_stage.rs1_s;
    assign monitor_rs2_addr  = writeback_stage.rs2_s;
    assign monitor_rs1_rdata = writeback_stage.rs1_v;
    assign monitor_rs2_rdata = writeback_stage.rs2_v;
    assign monitor_rd_addr   = writeback_stage.rd_s;
    assign monitor_rd_wdata  = writeback_stage.rd_v_wb;
    assign monitor_pc_rdata  = writeback_stage.pc;
    assign monitor_pc_wdata  = writeback_stage.pc_next;

    assign monitor_mem_addr  = writeback_stage.mem_addr;
    assign monitor_mem_rmask = writeback_stage.mem_rmask; //change r mask
    assign monitor_mem_wmask = writeback_stage.mem_wmask;
    assign monitor_mem_rdata = writeback_stage.mem_rdata;
    assign monitor_mem_wdata = writeback_stage.mem_wdata;
*/


endmodule
