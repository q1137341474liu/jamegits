
module control (
	input logic [31:0] instr,
	//input logic [6:0] opcode,
	//input logic [6:0] funct7,
    	//input logic [2:0] funct3,
    	output logic regwrite,
	output logic alu_select,
	output logic memwrite,
	output logic wb_src,
	//output logic branch,
    	output logic [1:0]ext,
    	output logic [3:0]alu_control,
	//output logic pc_src,
	output logic memread,
	output logic [2:0]funct3_d,
	output logic [6:0]funct7_d,
	output logic [4:0] rs1_d,
	output logic [4:0] rs2_d,
	output logic [4:0] rd_d,
	output logic unsign_d,
	output logic commit_d
	//output logic memtoreg

);
logic [6:0] opcode;
logic [2:0] funct3;
logic [6:0] funct7;
assign opcode = instr[6:0];
assign funct7 = instr[31:25];
assign funct3 = instr[14:12];

assign funct3_d = instr[14:12];
assign funct7_d = instr[31:25];
assign rs1_d = instr[19:15];
assign rs2_d = instr[24:20];

always_comb begin 
	if(opcode != 7'b1100011 & opcode != 7'b0100011) begin
	rd_d = instr[11:7];
	end else begin
	rd_d = '0;
	end
end

always_comb begin
	if((instr[14:12] == 3'b100) | (instr[14:12] == 3'b101) ) begin
		unsign_d = 1'b1;
	end else begin
		unsign_d = 1'b0;
	end
end

    opdecoder Opdecoder(
                .opcode(opcode),
                .regwrite_d(regwrite),
                .ext_d(ext),
                .memwrite_d(memwrite),
                .wb_src_d(wb_src),
                //.branch_d(branch),
                .alu_select_d(alu_select),
		//.pc_src_d(pc_src),
		.memread_d(memread),
		.commit(commit_d)
		//.memtoreg(memtoreg)
    );

	alu_decoder aludecoder (

	.funct3(funct3),
	.funct7(funct7),
	.opcode(opcode),
	.alu_control(alu_control)

	
);


endmodule
