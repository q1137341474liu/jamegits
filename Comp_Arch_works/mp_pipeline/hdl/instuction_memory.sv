module Instruction_Memory(
	//input logic clk,
  	input logic rst,
  	input logic [31:0] pc,
	input logic [31:0] imem_rdata,
	//input logic imem_resp,
	output logic [31:0] instr_f,
	output logic [31:0] imem_addr,
	output logic [3:0] imem_rmask
	//output logic stall

);
always_comb begin
	if (rst) begin
	imem_rmask = '0;
	end else begin
	imem_rmask = 4'b1111;
	end
end

/*
always_comb begin
	if (imem_resp == 1) begin
	instr_f = imem_rdata;
	//stall = 1'b0;
	end else begin
	//stall = 1'b1;
	instr = '0;
	end
end
*/


assign instr_f = imem_rdata;
assign imem_addr = pc;

endmodule
