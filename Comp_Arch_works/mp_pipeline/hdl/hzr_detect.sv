module stall
import rv32i_types::*;
(   
    input logic clk,
    input logic rst,
    input logic imem_need,
    input logic dmem_need,
    input logic imem_resp,
    input logic dmem_resp,
    output logic go
);


always_ff @(posedge clk) begin
	if(rst) begin
	go <= '0;
	end  if (imem_need == 1'b1 && imem_resp == 1'b1 && dmem_need  == '0) begin
	go <= 1'b1;
	end  if (dmem_need == 1'b1 && dmem_resp == 1'b1 && imem_need == '0) begin
	go <= 1'b1;
	end  if (imem_need == 1'b1 && imem_resp == 1'b1 && dmem_need  == 1'b1 && imem_need == 1'b1) begin
	go <= 1'b1;
	end
end
endmodule


