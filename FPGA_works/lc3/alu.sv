module ALU(input [15:0] IR, SR1_Data, SR2_Data,
							input [1:0]	ALUK,
							input SR2MUX,
							output logic [15:0] ALU);
			
	logic [15:0] B_In;
	
	always_comb begin
		if (SR2MUX)
			B_In = {{11{IR[4]}}, IR[4:0]};
		else
			B_In = SR2_Data;
		
		unique case	(ALUK)
			2'b00		:	ALU = SR1_Data + B_In;
			2'b01		:	ALU = SR1_Data & B_In;
			2'b10		:	ALU = ~SR1_Data;
			2'b11		:	ALU = SR1_Data;
		endcase
	end
	
endmodule
