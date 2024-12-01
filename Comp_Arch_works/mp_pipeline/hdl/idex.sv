module idex (
	input logic clk,
	input logic rst,
	input logic [31:0] instr_d,
	input logic [31:0] pc_d,
	input logic [31:0] pc_4_d,
	input logic [31:0] imm_ext_d,
	input logic [2:0] funct3_d,
	input logic [6:0] funct7_d,
	input logic [4:0] rd_d,
	input logic [4:0] rs1_d,
	input logic [4:0] rs2_d,
	input logic unsign_d,
	input logic stall,
	input logic commit_d,
	output logic [31:0] instr_e,
	output logic [31:0] pc_e,
	output logic [31:0] pc_4_e,
	output logic [31:0] imm_ext_e,
	output logic [2:0] funct3_e,
	output logic [6:0] funct7_e,
	output logic [4:0] rd_e,
	output logic [4:0] rs1_e,
	output logic [4:0] rs2_e,
	output logic unsign_e,

//control bits
	input logic regwrite_d,
	input logic alu_select_d,
	input logic memwrite_d,
	input logic wb_src_d,
	//input logic branch_d,
	input logic [3:0] alu_control_d,
	//input logic pc_src_d,
	input logic memread_d,
	output logic regwrite_e,
	output logic alu_select_e,
	output logic memwrite_e,
	output logic wb_src_e,
	//output logic branch_e,
	output logic [3:0] alu_control_e,
	//output logic pc_src_e,
	output logic memread_e,
	output logic commit_e

);

logic [31:0] instr_d_in;
logic [31:0] pc_d_in;
logic [31:0] pc_4_d_in;
logic [31:0] imm_ext_d_in;
logic [2:0] funct3_d_in;
logic [6:0] funct7_d_in;
logic commit_in;
logic [4:0] rd_d_in;
logic [4:0] rs1_d_in;
logic [4:0] rs2_d_in;
logic unsign_d_in;
logic regwrite_d_in;
logic alu_select_d_in;
logic memwrite_d_in;
logic wb_src_d_in;
	//input logic branch_d,
logic [3:0] alu_control_d_in;
	//input logic pc_src_d,
logic memread_d_in;

logic [31:0] instr_d_out;
logic [31:0] pc_d_out;
logic [31:0] pc_4_d_out;
logic [31:0] imm_ext_d_out;
logic [2:0] funct3_d_out;
logic [6:0] funct7_d_out;
logic commit_out;
logic [4:0] rd_d_out;
logic [4:0] rs1_d_out;
logic [4:0] rs2_d_out;
logic unsign_d_out;
logic regwrite_d_out;
logic alu_select_d_out;
logic memwrite_d_out;
logic wb_src_d_out;
	//input logic branch_d,
logic [3:0] alu_control_d_out;
	//input logic pc_src_d,
logic memread_d_out;

logic [31:0] instr_d_stall;
logic [31:0] pc_d_stall;
logic [31:0] pc_4_d_stall;
logic [31:0] imm_ext_d_stall;
logic [2:0] funct3_d_stall;
logic [6:0] funct7_d_stall;
logic commit_stall;
logic [4:0] rd_d_stall;
logic [4:0] rs1_d_stall;
logic [4:0] rs2_d_stall;
logic unsign_d_stall;
logic regwrite_d_stall;
logic alu_select_d_stall;
logic memwrite_d_stall;
logic wb_src_d_stall;
	//input logic branch_d,
logic [3:0] alu_control_d_stall;
	//input logic pc_src_d,
logic memread_d_stall;


always_ff @(posedge clk) begin
	if(rst) begin
	pc_d_out <= '0;
	pc_4_d_out<= '0;
	instr_d_out <= '0;
	imm_ext_d_out <= '0;
	funct3_d_out <= '0;
	funct7_d_out <= '0;
	rd_d_out <= '0;	
	rs1_d_out <= '0;
	rs2_d_out <= '0;
	regwrite_d_out <= '0;
	alu_select_d_out <= '0;
	memwrite_d_out <= '0;
	wb_src_d_out <= '0;
	//branch_e <= '0;
	alu_control_d_out <= '0;
	//pc_src_e <= '0;
	memread_d_out <= '0;
	unsign_d_out <= '0;
	commit_out <= '0;
	end else begin
	pc_d_out <= pc_d_in;
	pc_4_d_out <= pc_4_d_in;
	instr_d_out <= instr_d_in;
	imm_ext_d_out <= imm_ext_d_in;
	funct3_d_out <= funct3_d_in;
	funct7_d_out <= funct7_d_in;
	rd_d_out <= rd_d_in;	
	rs1_d_out <= rs1_d_in;
	rs2_d_out <= rs2_d_in;
	regwrite_d_out <= regwrite_d_in;
	alu_select_d_out <= alu_select_d_in;
	memwrite_d_out <= memwrite_d_in;
	wb_src_d_out<= wb_src_d_in;
	//branch_e <= branch_d;
	alu_control_d_out <= alu_control_d_in;
	//pc_src_e <= pc_src_d;
	memread_d_out <= memread_d_in;
	unsign_d_out <= unsign_d_in;
	commit_out <= commit_in;
	end
end

always_comb begin
instr_d_stall = instr_d_out;
pc_d_stall = pc_d_out;
pc_4_d_stall = instr_d_out;
imm_ext_d_stall = imm_ext_d_out;
funct3_d_stall = funct3_d_out;
funct7_d_stall = funct7_d_out;
commit_stall = commit_out;
rd_d_stall = rd_d_out;
rs1_d_stall= rs1_d_out;
rs2_d_stall = rs2_d_out;
unsign_d_stall = unsign_d_out;
regwrite_d_stall = regwrite_d_out;
alu_select_d_stall = alu_select_d_out;
memwrite_d_stall = memwrite_d_out;
wb_src_d_stall = wb_src_d_out;
	//input logic branch_d,
alu_control_d_stall = alu_control_d_out;
	//input logic pc_src_d,
memread_d_stall = memread_d_out;

instr_e = instr_d_out;
pc_e = pc_d_out;
pc_4_e = instr_d_out;
imm_ext_e = imm_ext_d_out;
funct3_e = funct3_d_out;
funct7_e = funct7_d_out;
commit_e = commit_out;
rd_e = rd_d_out;
rs1_e= rs1_d_out;
rs2_e = rs2_d_out;
unsign_e = unsign_d_out;
regwrite_e = regwrite_d_out;
alu_select_e = alu_select_d_out;
memwrite_e = memwrite_d_out;
wb_src_e = wb_src_d_out;
	//input logic branch_d,
alu_control_e = alu_control_d_out;
	//input logic pc_src_d,
memread_e = memread_d_out;

	if(stall) begin
instr_d_in = instr_d_stall;
pc_d_in = pc_d_stall;
pc_4_d_in = instr_d_stall;
imm_ext_d_in = imm_ext_d_stall;
funct3_d_in = funct3_d_stall;
funct7_d_in = funct7_d_stall;
commit_in = commit_stall;
rd_d_in = rd_d_stall;
rs1_d_in= rs1_d_stall;
rs2_d_in = rs2_d_stall;
unsign_d_in = unsign_d_stall;
regwrite_d_in = regwrite_d_stall;
alu_select_d_in = alu_select_d_stall;
memwrite_d_in = memwrite_d_stall;
wb_src_d_in = wb_src_d_stall;
	//input logic branch_d,
alu_control_d_in = alu_control_d_stall;
	//input logic pc_src_d,
memread_d_in = memread_d_stall;

instr_d_stall = instr_d_out;
pc_d_stall = pc_d_out;
pc_4_d_stall = instr_d_out;
imm_ext_d_stall = imm_ext_d_out;
funct3_d_stall = funct3_d_out;
funct7_d_stall = funct7_d_out;
commit_stall = commit_out;
rd_d_stall = rd_d_out;
rs1_d_stall= rs1_d_out;
rs2_d_stall = rs2_d_out;
unsign_d_stall = unsign_d_out;
regwrite_d_stall = regwrite_d_out;
alu_select_d_stall = alu_select_d_out;
memwrite_d_stall = memwrite_d_out;
wb_src_d_stall = wb_src_d_out;
	//input logic branch_d,
alu_control_d_stall = alu_control_d_out;
	//input logic pc_src_d,
memread_d_stall = memread_d_out;

	end else begin
instr_d_in = instr_d;
pc_d_in = pc_d;
pc_4_d_in = pc_4_d;
imm_ext_d_in = imm_ext_d;
funct3_d_in = funct3_d;
funct7_d_in = funct7_d;
commit_in = commit_d;
rd_d_in = rd_d;
rs1_d_in= rs1_d;
rs2_d_in = rs2_d;
unsign_d_in = unsign_d;
regwrite_d_in = regwrite_d;
alu_select_d_in = alu_select_d;
memwrite_d_in = memwrite_d;
wb_src_d_in = wb_src_d;
	//input logic branch_d,
alu_control_d_in = alu_control_d;
	//input logic pc_src_d,
memread_d_in = memread_d;

instr_e = instr_d_out;
pc_e = pc_d_out;
pc_4_e = instr_d_out;
imm_ext_e = imm_ext_d_out;
funct3_e = funct3_d_out;
funct7_e = funct7_d_out;
commit_e = commit_out;
rd_e = rd_d_out;
rs1_e= rs1_d_out;
rs2_e = rs2_d_out;
unsign_e = unsign_d_out;
regwrite_e = regwrite_d_out;
alu_select_e = alu_select_d_out;
memwrite_e = memwrite_d_out;
wb_src_e = wb_src_d_out;
	//input logic branch_d,
alu_control_e = alu_control_d_out;
	//input logic pc_src_d,
memread_e = memread_d_out;
	end

	
end


endmodule
