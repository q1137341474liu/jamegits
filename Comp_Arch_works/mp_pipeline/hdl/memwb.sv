module memwb (
	input logic clk,
	input logic rst,
	input logic [4:0] rs1_m,
	input logic [4:0] rs2_m,
	input logic [31:0] rs1_v_m,	
	input logic [31:0] rs2_v_m,
	input logic [31:0] pc_m,
	input logic [31:0] pc_4_m,
	//input logic [31:0] pc_imm_m,
	//input logic [31:0] imm_m,
	input logic [31:0] alu_result_m,
	input logic [31:0] instr_m,
	input logic [4:0] rd_m,
	//input logic rw_sel_m,
	input logic memtoreg_m,
	input logic regwrite_m,
	output logic [31:0] pc_wb,
	output logic [31:0] pc_4_wb,
	//output logic [31:0] pc_imm_wb,
	//output logic [31:0] imm_wb,
	output logic [31:0] alu_result_wb,
	output logic [4:0] rd_wb,
	//output logic rw_sel_wb,
	output logic memtoreg_wb,
	output logic regwrite_wb,
	output logic [31:0] instr_wb,
	output logic [4:0] rs1_wb,
	output logic [4:0] rs2_wb,
	output logic [31:0] rs1_v_wb,	
	output logic [31:0] rs2_v_wb,
	input logic [31:0] rdata_m,
	output logic [31:0] rdata_wb
);

always_ff @(posedge clk) begin
	if(rst) begin
	pc_wb <= '0;
	pc_4_wb <= '0;
	//pc_imm_wb <= '0;
	//imm_wb <= '0;
	alu_result_wb <= '0;
	rd_wb <= '0;
	//rw_sel_wb <= 0';
	memtoreg_wb <= '0;
	regwrite_wb <= '0;
	instr_wb <= '0;
	rs1_wb <= '0;
	rs2_wb <= '0;
	rs1_v_wb <= '0;
	rs2_v_wb <= '0;
	rdata_wb <= '0;
	end else begin
	pc_wb <= pc_m;
	pc_4_wb <= pc_4_m;
	//pc_imm_wb <= pc_imm_m;
	//imm_wb <= imm_m;
	alu_result_wb <= alu_result_m;
	rd_wb <= rd_m;
	//rw_sel_wb <= rw_sel_m;
	memtoreg_wb <= memtoreg_m;
	regwrite_wb <= regwrite_m;	
	instr_wb <= instr_m;
	rs1_wb <= rs1_m;
	rs2_wb <= rs2_m;
	rs1_v_wb <= rs1_v_m;
	rs2_v_wb <= rs2_v_m;
	rdata_wb <= rdata_m;
	end
end
endmodule
