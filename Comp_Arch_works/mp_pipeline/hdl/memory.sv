module memory 
import rv32i_types::*;
(
	input clk,
	input rst,
	input logic go,
    	input   ex_mem_stage_reg_t ex_mem_stage_reg,
   	output  mem_wb_stage_reg_t mem_wb_stage_reg,
	output logic dmem_need,
	output logic [31:0] dmem_addr,
	output logic [31:0] dmem_wdata,
	output logic [3:0] dmem_wmask,
	output logic [3:0] dmem_rmask,
    input   logic   [31:0]  dmem_rdata,
	input logic dmem_resp
);


            logic   [63:0]  order;
            logic   valid;
            logic   [31:0]  pc;
            logic   [31:0]  pc_next;
  

            logic   [31:0]  inst;
            logic   [6:0]   opcode;
assign opcode = inst[6:0];
            logic   [31:0]  u_imm;
            logic   [31:0]  rs1_v;
            logic   [31:0]  rs2_v;
            logic           regf_we;
            logic   [4:0]  rs1_s;
            logic   [4:0]  rs2_s;
            logic   [4:0]  rd_s;
    control_mem_t    control_mem;
    control_wb_t     control_wb;

	logic [31:0] alu_out;
	logic [31:0] branch;
	logic [31:0] load_type;
    	logic [31:0] mem_rdata;
	logic [31:0] dmem_rdata_reg;
	logic [31:0] mem_addr;

    always_comb begin 
        order = ex_mem_stage_reg.order;
        valid = ex_mem_stage_reg.valid;
	pc_next = ex_mem_stage_reg.pc_next;
        pc = ex_mem_stage_reg.pc;
        inst = ex_mem_stage_reg.inst;
        u_imm = ex_mem_stage_reg.u_imm;
	control_mem = ex_mem_stage_reg.control_mem;
	control_wb = ex_mem_stage_reg.control_wb;
	alu_out = ex_mem_stage_reg.alu_out;
	branch = ex_mem_stage_reg.branch;
	mem_addr = ex_mem_stage_reg.mem_addr;

        rs1_v = ex_mem_stage_reg.rs1_v;
        rs2_v = ex_mem_stage_reg.rs2_v;
        rs1_s = ex_mem_stage_reg.rs1_s;
        rs2_s = ex_mem_stage_reg.rs2_s;
        rd_s = ex_mem_stage_reg.rd_s;
    end 

always_comb begin
        dmem_addr = '0;
        dmem_rmask = '0;
        dmem_wmask = '0;
        dmem_wdata =  '0;
        dmem_need = '0;
	if((opcode == op_b_load || opcode == op_b_store) && go == 1'b1) begin
        dmem_addr = ex_mem_stage_reg.mem_addr & 32'hfffffffc;
        dmem_rmask = ex_mem_stage_reg.mem_rmask;
        dmem_wmask = ex_mem_stage_reg.mem_wmask;
        dmem_wdata =  ex_mem_stage_reg.mem_wdata;
        dmem_need = 1'b1;	
	end
end

    always_ff @ (posedge clk) begin 
        if (rst == 1) begin 
            dmem_rdata_reg <= '0;
        end else if (dmem_resp) begin 
            dmem_rdata_reg <= dmem_rdata;
        end 
    end 

    always_comb begin 
        mem_rdata = '0;
	load_type = 'x;

        if (control_mem.memread == '1 && go == '1 && dmem_resp == '1) begin
            case (control_mem.load_ops)
                load_lb: begin 
                    load_type = {{24{dmem_rdata[7 +8 * mem_addr[1:0]]}}, dmem_rdata[8 * mem_addr[1:0] +: 8 ]};
                    mem_rdata = dmem_rdata;
                end
                load_lbu: begin 
                    load_type = {{24{1'b0}}                          , dmem_rdata[8 * mem_addr[1:0] +: 8 ]};
                    mem_rdata = dmem_rdata;
                end
                load_lh: begin 
                    load_type = {{16{dmem_rdata[15+16*mem_addr[1]  ]}}, dmem_rdata[16 * mem_addr[1]   +: 16]};
                    mem_rdata = dmem_rdata;
                end 
                load_lhu: begin 
                    load_type = {{16{1'b0}}                          , dmem_rdata[16*mem_addr[1]   +: 16]};
                    mem_rdata = dmem_rdata;
                end
                load_lw: begin 
                    load_type = dmem_rdata;
                    mem_rdata = dmem_rdata;
                end
            endcase
        end else if (control_mem.memread == '1 && go == '1) begin 
            case (control_mem.load_ops)
                load_lb: begin 
                    load_type = {{24{dmem_rdata_reg[7 +8 * mem_addr[1:0]]}}, dmem_rdata_reg[8 * mem_addr[1:0] +: 8 ]};
                    mem_rdata = dmem_rdata_reg;
                end
                load_lbu: begin 
                    load_type = {{24{1'b0}}                          , dmem_rdata_reg[8 * mem_addr[1:0] +: 8 ]};
                    mem_rdata = dmem_rdata_reg;
                end
                load_lh: begin 
                    load_type = {{16{dmem_rdata_reg[15+16*mem_addr[1]  ]}}, dmem_rdata_reg[16 * mem_addr[1]   +: 16]};
                    mem_rdata = dmem_rdata_reg;
                end 
                load_lhu: begin 
                    load_type = {{16{1'b0}}                          , dmem_rdata_reg[16*mem_addr[1]   +: 16]};
                    mem_rdata = dmem_rdata_reg;
                end
                load_lw: begin 
                    load_type = dmem_rdata_reg;
                    mem_rdata = dmem_rdata_reg;
                end
            endcase
        end 
    end


    always_comb begin 
        mem_wb_stage_reg.order = order;
        mem_wb_stage_reg.valid = valid;
        mem_wb_stage_reg.pc = pc;
	mem_wb_stage_reg.pc_next = pc_next;
        mem_wb_stage_reg.inst = inst;
        mem_wb_stage_reg.u_imm = u_imm;

	mem_wb_stage_reg.control_mem = control_mem;
	mem_wb_stage_reg.control_wb = control_wb;
	mem_wb_stage_reg.alu_out = alu_out;
	mem_wb_stage_reg.branch = branch;
        mem_wb_stage_reg.rs1_v = rs1_v;
        mem_wb_stage_reg.rs2_v = rs2_v; 
        mem_wb_stage_reg.rs1_s = rs1_s;
        mem_wb_stage_reg.rs2_s = rs2_s;
        mem_wb_stage_reg.rd_s = rd_s;

        mem_wb_stage_reg.mem_addr = ex_mem_stage_reg.mem_addr;
        mem_wb_stage_reg.mem_rmask = ex_mem_stage_reg.mem_rmask;
        mem_wb_stage_reg.mem_wmask = ex_mem_stage_reg.mem_wmask;
        mem_wb_stage_reg.mem_wdata = ex_mem_stage_reg.mem_wdata;
	mem_wb_stage_reg.mem_rdata = mem_rdata;
	mem_wb_stage_reg.load_type = load_type;
    end 


endmodule





