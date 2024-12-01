module writeback 
import rv32i_types::*;
(
	input logic go,
	input mem_wb_stage_reg_t mem_wb_stage_reg,

	output logic [31:0] rd_v_wb,
	output logic [4:0] rd_s_wb,
	output logic regf_we_wb

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
    	logic [31:0] mem_addr;
	logic [31:0] mem_rdata;
    	logic [31:0] mem_wdata;
    	logic [3:0]  mem_rmask;
    	logic [3:0]  mem_wmask;

	logic [31:0] load_type;

    always_comb begin 
        order = mem_wb_stage_reg.order;
        pc = mem_wb_stage_reg.pc;
	pc_next = mem_wb_stage_reg.pc_next;
        inst = mem_wb_stage_reg.inst;
        u_imm = mem_wb_stage_reg.u_imm;
	control_mem = mem_wb_stage_reg.control_mem;
	control_wb = mem_wb_stage_reg.control_wb;
	alu_out = mem_wb_stage_reg.alu_out;
	branch = mem_wb_stage_reg.branch;
        rs1_v = mem_wb_stage_reg.rs1_v;
        rs2_v = mem_wb_stage_reg.rs2_v;
        rs1_s = mem_wb_stage_reg.rs1_s;
        rs2_s = mem_wb_stage_reg.rs2_s;
        rd_s = mem_wb_stage_reg.rd_s;
        mem_addr = mem_wb_stage_reg.mem_addr;
        mem_rmask = mem_wb_stage_reg.mem_rmask;
        mem_wmask = mem_wb_stage_reg.mem_wmask;
        mem_wdata =  mem_wb_stage_reg.mem_wdata;
	mem_rdata = mem_wb_stage_reg.mem_rdata;
	load_type = mem_wb_stage_reg.load_type;
	if(control_mem.memwrite == 1) begin
	rd_s = '0;
	end if (inst[6:0] == op_b_br) begin
	rd_s = '0;
	end if (control_mem.memread) begin
	rs2_s = '0;
	end
    end 


    // write back
    always_comb begin 
        rd_v_wb = '0;
        if (mem_wb_stage_reg.valid) begin
            case (control_wb.regf_mux)
                alu_out_wb: begin
                    rd_v_wb = alu_out;
                end 
                branch_wb:   begin 
                    rd_v_wb = branch;
                end 
                u_imm_wb:   begin       
                    rd_v_wb = u_imm;
                end 
                lw_wb:  begin             
                    rd_v_wb = load_type;
                end
                lb_wb:  begin            
                    rd_v_wb = load_type;
                end
                lbu_wb: begin            
                    rd_v_wb = load_type;
                end
                lh_wb:  begin            
                    rd_v_wb = load_type;
                end
                lhu_wb: begin             
                    rd_v_wb = load_type;
                end
                pc_4_wb: begin
		    rd_v_wb = pc_next;
		end
            endcase
        end 
    end 

    always_comb begin
        rd_s_wb = rd_s;
        regf_we_wb = control_wb.regf_we;
        valid = '0;
        if (mem_wb_stage_reg.valid && go) begin
            valid = '1;
        end
    end 
endmodule
