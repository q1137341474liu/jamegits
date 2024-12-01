module mux4(input logic 	[15:0]	A, B, C, D,
					input logic 	[1:0] 	Select,
					output logic 	[15:0] 	out);
					
					
					always_comb
					begin
							case(Select)
								2'b00	:	out = A;
								2'b01	:	out = B;
								2'b10	:	out = C;
								2'b11	: 	out = D; 
							endcase
					end
endmodule



module mux2(input logic 	[15:0]	A, B,
					input logic 			 	Select,
					output logic 	[15:0] 	out);
					
					
					always_comb
					begin
							case(Select)
								1'b0	:	out = A;
								1'b1	:	out = B; 
				
							endcase
					end
endmodule

