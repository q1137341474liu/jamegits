module decode
import rv32i_types::*;
(
	input logic clk,
	input logic rst,
	input logic imem_resp,
	input logic [31:0] imem_rdata,
	input logic [4:0] rd_s_wb,
	input logic [31:0] rd_v_wb,
	input logic regf_we_wb,
	input logic stall,
	input logic flush,
	input logic go,
	//input logic flush_delay,	
	output logic [4:0] rs1_s_d,
	output logic [4:0] rs2_s_d,
		
	input if_id_stage_reg_t if_id_stage_reg,
	output id_ex_stage_reg_t id_ex_stage_reg,
//forwarding rs value
    input   rs1_forward_id_t         rs1_forward_id,
    input   rs2_forward_id_t         rs2_forward_id
); 
            logic   [63:0]  order;
            logic   valid;
            logic   [31:0]  pc;
            logic   [31:0]  pc_next;
  

            logic   [31:0]  inst;
            logic   [2:0]   funct3;
            logic   [6:0]   funct7;
            logic   [6:0]   opcode;
            logic   [31:0]  i_imm;
            logic   [31:0]  s_imm;
            logic   [31:0]  b_imm;
            logic   [31:0]  u_imm;
            logic   [31:0]  j_imm;
            logic   [31:0]  rs1_v_d;
            logic   [31:0]  rs2_v_d;

	    //logic   [4:0]   shamt;
            logic           regf_we;
            logic   [4:0]  rd_s;
	logic [31:0] imem_rdata_reg;
//some control signals to be implemented
    control_ex_t     control_ex;
    control_mem_t    control_mem;
    control_wb_t     control_wb;


//if imem_resp not ready, do not produce new instruction
    always_ff @ (posedge clk) begin 
        if (rst) begin
            imem_rdata_reg <= '0;
        end else if (imem_resp) begin 
            imem_rdata_reg <= imem_rdata;
        end 
    end 


//instr decode
    always_comb begin
        valid = 1'b0;
	pc = 'x;
	pc_next = 'x;
	order = '0;
	funct3 = 'x;
	funct7 = 'x;
	opcode = 'x;
	i_imm = '0;
	s_imm = '0;
	b_imm = '0;
	j_imm = '0;
	u_imm = '0;
	rs1_s_d = 'x;
	rs2_s_d = 'x;
	rd_s = 'x;
	inst = 'x;	

	if(imem_resp == 1'b1 && go == 1'b1) begin
		inst = imem_rdata;
		valid = if_id_stage_reg.valid;
		pc = if_id_stage_reg.pc;
		pc_next = if_id_stage_reg.pc_next;
		order = if_id_stage_reg.order;	
		funct3 = imem_rdata[14:12];
		funct7 = imem_rdata[31:25];
		opcode = imem_rdata[6:0];
		i_imm = {{21{imem_rdata[31]}}, imem_rdata[30:20]};
		s_imm = {{21{imem_rdata[31]}}, imem_rdata[30:25], imem_rdata[11:7]};
		b_imm = {{20{imem_rdata[31]}}, imem_rdata[7], imem_rdata[30:25], imem_rdata[11:8], 1'b0};
		j_imm = {{12{imem_rdata[31]}}, imem_rdata[19:12], imem_rdata[20], imem_rdata[30:21], 1'b0};
		u_imm = {imem_rdata[31:12], 12'h000};
		rs1_s_d = imem_rdata[19:15];
		rs2_s_d = imem_rdata[24:20];
		rd_s = imem_rdata[11:7];
	end else if (imem_resp == 1'b0 && go == 1'b1) begin
		inst = imem_rdata_reg;
		valid = if_id_stage_reg.valid;
		pc = if_id_stage_reg.pc;
		pc_next = if_id_stage_reg.pc_next;
		order = if_id_stage_reg.order;
		funct3 = imem_rdata_reg[14:12];
		funct7 = imem_rdata_reg[31:25];
		opcode = imem_rdata_reg[6:0];
		i_imm = {{21{imem_rdata_reg[31]}}, imem_rdata_reg[30:20]};
		s_imm = {{21{imem_rdata_reg[31]}}, imem_rdata_reg[30:25], imem_rdata_reg[11:7]};
		b_imm = {{20{imem_rdata_reg[31]}}, imem_rdata_reg[7], imem_rdata_reg[30:25], imem_rdata_reg[11:8], 1'b0};
		j_imm = {{12{imem_rdata_reg[31]}}, imem_rdata_reg[19:12], imem_rdata_reg[20], imem_rdata_reg[30:21], 1'b0};
		u_imm = {imem_rdata_reg[31:12], 12'h000};
		rs1_s_d = imem_rdata_reg[19:15];
		rs2_s_d = imem_rdata_reg[24:20];
		rd_s = imem_rdata_reg[11:7];		
	end 
end

//regfile
    regfile regfile(
        .clk(clk),
        .rst(rst),
        .regf_we(regf_we_wb),
        .rd_s(rd_s_wb),
        .rd_v(rd_v_wb),
        .rs1_s(rs1_s_d),
        .rs2_s(rs2_s_d),
        .rs1_v(rs1_v_d),
        .rs2_v(rs2_v_d),

        .rs1_forward_id(rs1_forward_id),
        .rs2_forward_id(rs2_forward_id)
    );

always_comb begin

//5 ex controls, 4 mem controls and 2 write back controls
	control_ex.alumux1_sel = 'x;
	control_ex.alumux2_sel = 'x;
	control_ex.alu_ops = 'x;
	control_ex.branch_mux = 'x;
	control_ex.branch_f3 = 'x;
	control_mem.memread = 'x;
	control_mem.memwrite = 'x;
	control_mem.load_ops = 'x;
	control_mem.store_ops = 'x;
	control_wb.regf_we = 'x;
	control_wb.regf_mux = 'x;

//different opcode with different controls
	case(opcode)
		op_b_lui: begin
		control_wb.regf_we = 1'b1;
		control_wb.regf_mux = u_imm_wb;
		end

		op_b_auipc: begin
		control_wb.regf_we = 1'b1;
		control_wb.regf_mux = alu_out_wb;
		control_ex.alumux1_sel = pc_out_alu;
		control_ex.alumux2_sel = u_imm_alu;
		control_ex.alu_ops = alu_op_add;
		end

		op_b_jal: begin
		control_wb.regf_we = 1'b1;
		control_wb.regf_mux = pc_4_wb;
		end

		op_b_jalr: begin
		control_wb.regf_we = 1'b1;
		control_wb.regf_mux = pc_4_wb;		
		end

		op_b_br: begin
		control_ex.branch_mux = rs2_branch_ex;
		control_ex.branch_f3 = funct3;
		end

		op_b_load: begin
		control_wb.regf_we = 1'b1;
		control_ex.alumux1_sel = rs1_out_alu;
		control_ex.alumux2_sel = i_imm_alu;
		control_ex.alu_ops = alu_op_add;
		control_mem.load_ops = funct3;
		control_mem.memread = 1'b1;
		case(funct3)
		load_f3_lb: control_wb.regf_mux = lb_wb;
		load_f3_lh: control_wb.regf_mux = lh_wb;
		load_f3_lw: control_wb.regf_mux = lw_wb;
		load_f3_lbu: control_wb.regf_mux = lbu_wb;
		load_f3_lhu: control_wb.regf_mux = lhu_wb;
		endcase
		end

		op_b_store: begin
		//control_wb_regf_we = 1'b0;
		control_ex.alumux1_sel = rs1_out_alu;
		control_ex.alumux2_sel = s_imm_alu;
		control_ex.alu_ops = alu_op_add;
		control_mem.store_ops = funct3;
		control_mem.memwrite = 1'b1;			
		end

		op_b_imm: begin
		case(funct3)
		arith_f3_slt: begin
			control_ex.branch_mux = i_imm_branch_ex;
			control_ex.branch_f3 = branch_f3_blt;
			control_wb.regf_mux = branch_wb;
			control_wb.regf_we = 1'b1;
			end
		arith_f3_sltu: begin
			control_ex.branch_mux = i_imm_branch_ex;
			control_ex.branch_f3 = branch_f3_bltu;
			control_wb.regf_mux = branch_wb;
			control_wb.regf_we = 1'b1;
			end
		arith_f3_sr: begin
			if(funct7[5] == 1'b1) begin
			control_ex.alu_ops = alu_op_sra;
			control_wb.regf_mux = alu_out_wb;
			control_ex.alumux1_sel = rs1_out_alu;
			control_ex.alumux2_sel = i_imm_alu;
			control_wb.regf_we = 1'b1;
			end else if(funct7[5] == 1'b0) begin
			control_ex.alu_ops = alu_op_srl;
			control_wb.regf_mux = alu_out_wb;
			control_ex.alumux1_sel = rs1_out_alu;
			control_ex.alumux2_sel = i_imm_alu;
			control_wb.regf_we = 1'b1;
			end
		end
		default: begin
			control_ex.alu_ops = funct3;
			control_wb.regf_mux = alu_out_wb;
			control_ex.alumux1_sel = rs1_out_alu;
			control_ex.alumux2_sel =  i_imm_alu; 
			control_wb.regf_we = 1'b1;
		end
		endcase			
	end
		op_b_reg: begin
		case(funct3)
		arith_f3_slt: begin
			control_ex.branch_mux = rs2_branch_ex;
			control_ex.branch_f3 = branch_f3_blt;
			control_wb.regf_mux = branch_wb;
			control_wb.regf_we = 1'b1;
		end
		arith_f3_sltu: begin
			control_ex.branch_mux = rs2_branch_ex;
			control_ex.branch_f3 = branch_f3_bltu;
			control_wb.regf_mux = branch_wb;
			control_wb.regf_we = 1'b1;
		end
		arith_f3_sr: begin
			if(funct7[5] == 1'b1) begin
			control_ex.alu_ops = alu_op_sra;
			control_wb.regf_mux = alu_out_wb;
			control_ex.alumux1_sel = rs1_out_alu;
			control_ex.alumux2_sel = rs2_out_alu;
			control_wb.regf_we = 1'b1;
			end else if(funct7[5] == 1'b0) begin
			control_ex.alu_ops = alu_op_srl;
			control_wb.regf_mux = alu_out_wb;
			control_ex.alumux1_sel = rs1_out_alu;
			control_ex.alumux2_sel = rs2_out_alu;
			control_wb.regf_we = 1'b1;
			end
		end	
		arith_f3_add: begin
			if(funct7[5] == 1'b1) begin
			control_ex.alu_ops = alu_op_sub;
			control_wb.regf_mux = alu_out_wb;
			control_ex.alumux1_sel = rs1_out_alu;
			control_ex.alumux2_sel = rs2_out_alu;
			control_wb.regf_we = 1'b1;
			end else if(funct7[5] == 1'b0) begin
			control_ex.alu_ops = alu_op_add;
			control_wb.regf_mux = alu_out_wb;
			control_ex.alumux1_sel = rs1_out_alu;
			control_ex.alumux2_sel = rs2_out_alu;
			control_wb.regf_we = 1'b1;
			end
		end	
		default: begin
			control_ex.alu_ops = funct3;
			control_wb.regf_mux = alu_out_wb;
			control_ex.alumux1_sel = rs1_out_alu;
			control_ex.alumux2_sel = rs2_out_alu; 
			control_wb.regf_we = 1'b1;
		end
		endcase						
		end
	endcase
end


always_comb begin 
        id_ex_stage_reg.order = order;
        id_ex_stage_reg.valid = valid;
        id_ex_stage_reg.pc = pc;
        id_ex_stage_reg.pc_next = pc_next;
        id_ex_stage_reg.inst = inst;
        id_ex_stage_reg.rs1_v = rs1_v_d;
        id_ex_stage_reg.rs2_v = rs2_v_d;
        id_ex_stage_reg.rs1_s = rs1_s_d;
        id_ex_stage_reg.rs2_s = rs2_s_d;
        id_ex_stage_reg.rd_s = rd_s;
        id_ex_stage_reg.i_imm = i_imm;
        id_ex_stage_reg.s_imm = s_imm;
        id_ex_stage_reg.b_imm = b_imm;
        id_ex_stage_reg.u_imm = u_imm;
        id_ex_stage_reg.j_imm = j_imm;
      	id_ex_stage_reg.control_ex = control_ex;
        id_ex_stage_reg.control_mem = control_mem;
        id_ex_stage_reg.control_wb = control_wb;

        if (stall == 1'b1) begin 
            id_ex_stage_reg = '0;
        end else if (flush == 1'b1) begin 
            id_ex_stage_reg = '0;
        end 
end


endmodule
