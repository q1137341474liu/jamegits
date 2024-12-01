module alu_decoder (

    input logic [2:0]funct3,
    input logic [6:0]funct7,
    input logic [6:0]opcode,
    output logic [3:0]alu_control
);


always_comb begin
	if((opcode == 7'b0010011) & (funct3 == 3'b000)) begin
	alu_control = 4'b0000;
	end else if((opcode == 7'b0110011) &  (funct7 == 7'b0000000) & (funct3 == 3'b000)) begin
	alu_control = 4'b0000;
	end // addi and add

	else if((opcode == 7'b0110011) &  (funct7 == 7'b0100000) & (funct3 == 3'b000)) begin
	alu_control = 4'b1000; //subtraction
	end

	else if((opcode == 7'b0010011) & (funct3 == 3'b001)) begin
	alu_control = 4'b0001;
	end

	else if((opcode == 7'b0110011) &  (funct7 == 7'b0000000) & (funct3 == 3'b001)) begin
	alu_control = 4'b0001;
	end //sll

	else if((opcode == 7'b0010011) & (funct3 == 3'b010)) begin
	alu_control = 4'b0010;
	end
	else if((opcode == 7'b0110011) &  (funct7 == 7'b0000000) & (funct3 == 3'b010)) begin
	alu_control = 4'b0010;
	end //slt

	else if((opcode == 7'b0010011) & (funct3 == 3'b011)) begin
	alu_control = 4'b0011;
	end
	else if((opcode == 7'b0110011) &  (funct7 == 7'b0000000) & (funct3 == 3'b011)) begin
	alu_control = 4'b0011;
	end //sltu

	else if((opcode == 7'b0010011) & (funct3 == 3'b100)) begin
	alu_control = 4'b0100;
	end
	else if((opcode == 7'b0110011) &  (funct7 == 7'b0000000) & (funct3 == 3'b100)) begin
	alu_control = 4'b0100;
	end //xor

	else if((opcode == 7'b0010011) &  (funct7 == 7'b0000000) & (funct3 == 3'b101)) begin
	alu_control = 4'b0101;
	end
	else if((opcode == 7'b0010011) &  (funct7 == 7'b0100000) & (funct3 == 3'b101)) begin
	alu_control = 4'b1101;
	end
	else if((opcode == 7'b0110011) &  (funct7 == 7'b0000000) & (funct3 == 3'b101)) begin
	alu_control = 4'b0101;
	end
	else if((opcode == 7'b0110011) &  (funct7 == 7'b0100000) & (funct3 == 3'b101)) begin
	alu_control = 4'b1101;
	end

	else if((opcode == 7'b0010011) & (funct3 == 3'b110)) begin
	alu_control = 4'b0110;
	end
	else if((opcode == 7'b0110011) &  (funct7 == 7'b0000000) & (funct3 == 3'b110)) begin
	alu_control = 4'b0110;
	end

	else if((opcode == 7'b0010011) & (funct3 == 3'b111)) begin
	alu_control = 4'b0111;
	end
	else if((opcode == 7'b0110011) &  (funct7 == 7'b0000000) & (funct3 == 3'b111)) begin
	alu_control = 4'b0111;
	end

	else if(opcode == 7'b0110111) begin
	alu_control = 4'b1001; //lui
	end
	
	else if (opcode == 7'b0100011) begin
	alu_control = 4'b0000;
	end

	else if (opcode == 7'b0000011) begin
	alu_control = 4'b0000;
	end

	else begin
	alu_control = 4'b1111;
	end
end
endmodule
