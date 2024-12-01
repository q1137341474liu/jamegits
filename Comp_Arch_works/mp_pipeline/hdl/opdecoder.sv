module opdecoder (
	input logic [6:0] opcode,
	output logic regwrite_d,
	output logic alu_select_d,
	output logic memwrite_d,
	output logic memread_d,
	//output logic branch_d,
	output logic wb_src_d,
	output logic [1:0] ext_d,
	output logic commit
	//output logic pc_src_d
	//output logic memtoreg

);


always_comb begin
	if(opcode == 7'b0110111 | opcode == 7'b0010111 | opcode == 7'b0100011| opcode == 7'b1101111| opcode == 7'b1100111| opcode == 7'b1100011| opcode == 7'b0000011| opcode == 7'b0100011| opcode == 7'b0010011| opcode == 7'b0110011) begin
	commit = 1'b1;
	end else begin
	commit = 1'b0;
	end
end

always_comb begin
	if(opcode == 7'b0000011 | opcode == 7'b0110011 | opcode == 7'b0010011 | opcode == 7'b0110111) begin
	regwrite_d = 1'b1;
	end else begin
	regwrite_d = 1'b0;
	end
end


always_comb begin
	if(opcode == 7'b0100011) begin
	memwrite_d = 1'b1;
	end else begin
	memwrite_d = 1'b0;
	end
end


always_comb begin
	if(opcode == 7'b0000011) begin
	memread_d = 1'b1;
	end else begin
	memread_d = 1'b0;
	end
end


always_comb begin
	if(opcode == 7'b0000011 | opcode == 7'b0100011) begin
	wb_src_d = 1'b1;
	end else begin
	wb_src_d = 1'b0;
	end
end

always_comb begin
	if(opcode == 7'b0110111) begin
	ext_d = 2'b11;
	end else if (opcode == 7'b0010111) begin
	ext_d = 2'b01;
	end else begin
	ext_d = 2'b00;
	end
end

/*
always_comb begin
	if(opcode == 7'b1100011) begin
	//branch_d = 1'b1;
	pc_src_d = 1'b1;
	end else begin
	//branch_d = 1'b0;
	pc_src_d = 1'b0;
	end
end
*/

always_comb begin
	if(opcode == 7'b0110011 | opcode == 7'b1100011) begin
	alu_select_d = 1'b0;
	end else begin
	alu_select_d = 1'b1;
	end
end

/*
always_comb begin
	if(opcode == 7'b0000011) begin
	memtoreg = 1'b0;
	end else begin
	memtoreg = 1'b1;
	end
end
*/



endmodule
