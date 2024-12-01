module exmem (
	input logic clk,
	input logic rst,
	input logic [31:0] pc_e,	
	input logic [31:0] pc_4_e,	
	input logic [31:0] alu_result_e,
	input logic [31:0] rs1_v_e,	
	input logic [31:0] rs2_v_e,
	input logic [31:0] instr_e,
	input logic [2:0] funct3_e,
	input logic [6:0] funct7_e,
	input logic [4:0] rd_e,
	input logic [4:0] rs1_e,
	input logic [4:0] rs2_e,
	input logic regwrite_e,
	input logic memwrite_e,
	input logic wb_src_e,
	//input logic branch_e,
	input logic unsign_e,
	//input logic pc_src_e,
	input logic memread_e,
	output logic [31:0] pc_m,	
	output logic [31:0] pc_4_m,	
	output logic [31:0] alu_result_m,
	output logic [31:0] rs1_v_m,	
	output logic [31:0] rs2_v_m,
	output logic [31:0] instr_m,
	output logic [2:0] funct3_m,
	output logic [6:0] funct7_m,
	output logic [4:0] rd_m,
	output logic [4:0] rs1_m,
	output logic [4:0] rs2_m,
	output logic regwrite_m,
	output logic memwrite_m,
	output logic wb_src_m,
	//output logic branch_m,
	//output logic pc_src_m,
	output logic memread_m,
	output logic unsign_m

);

always_ff @(posedge clk) begin
	if(rst) begin
	pc_m <= '0;
	pc_4_m <= '0;
	instr_m <= '0;
	funct3_m <= '0;
	funct7_m <= '0;
	rd_m <= '0;	
	rs1_v_m <= '0;
	rs2_v_m <= '0;
	rs1_m <= '0;
	rs2_m <= '0;
	regwrite_m <= '0;
	memwrite_m <= '0;
	wb_src_m <= '0;
	//branch_m <= '0;
	unsign_m <= '0;
	//pc_src_m <= '0;
	memread_m <= '0;
	alu_result_m <= '0;
	end else begin
	pc_m <= pc_e;
	pc_4_m <= pc_4_e;
	instr_m <= instr_e;
	funct3_m <= funct3_e;
	funct7_m <= funct7_e;
	rd_m <= rd_e;	
	rs1_m <= rs1_e;
	rs2_m <= rs2_e;
	rs1_v_m <= rs1_v_e;
	rs2_v_m <= rs2_v_e;
	regwrite_m <= regwrite_e;
	memwrite_m <= memwrite_e;
	wb_src_m <= wb_src_e;
	//branch_m <= branch_e;
	unsign_m <= unsign_e;
	//pc_src_m <= pc_src_e;
	memread_m <= memread_e;
	alu_result_m <= alu_result_e;
	end
end

endmodule
