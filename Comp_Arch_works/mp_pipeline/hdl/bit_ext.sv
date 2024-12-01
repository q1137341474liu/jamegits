module bit_ext (
    input logic [31:0] instr,
    input logic [1:0] ext,
    output logic [31:0] imm_ext
);
always_comb begin
//u-type
	if (ext == 2'b11) begin
	imm_ext = {instr[31:12], 12'h000};
//i-type
	end else if (ext == 2'b01) begin
	imm_ext = {{20{instr[31]}},instr[31:20]};
//r-type or other cases
	end else begin
		if ((instr[6:0] == 7'b0010011) & ((instr[14:12] == 3'b101) | (instr[14:12] == 3'b001) )) begin
		imm_ext = {{27{1'b0}},instr[24:20]};
		end else if ((instr[6:0] == 7'b0010011) & ((instr[14:12] == 3'b000) | (instr[14:12] == 3'b010)| (instr[14:12] == 3'b011)| 
(instr[14:12] == 3'b100)| (instr[14:12] == 3'b110)| (instr[14:12] == 3'b111)) )begin
		imm_ext = {{20{1'b0}},instr[31:20]};
		end else if ((instr[6:0] == 7'b0000011)) begin
		imm_ext = {{20{1'b0}},instr[31:20]};
		end else if ((instr[6:0] == 7'b0100011)) begin
		imm_ext = {{20{1'b0}},instr[31:25],instr[11:7]};
		end else begin
		imm_ext = 32'b0;
		end
	end
end

endmodule
