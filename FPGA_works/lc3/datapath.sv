 module datapath 	( 	input Clk, Reset,
							input LD_MAR, LD_MDR, LD_IR, LD_BEN, LD_CC, LD_REG, LD_PC, LD_LED,
							input GateMDR, GateALU, GatePC, GateMARMUX,
							input SR2MUX, ADDR1MUX, MARMUX,
							input MIO_EN, DRMUX, SR1MUX,
							input [1:0] PCMUX, ADDR2MUX, ALUK,
							input [15:0] MDR_In,
							output logic [15:0] MAR, MDR, IR,
							output logic BEN,
							output logic [9:0] LED);

	logic [3:0] Tri;
	logic [15:0] BusData;
	assign Tri = {GateMDR, GateALU, GatePC, GateMARMUX};
	
	logic [15:0] ALU, PC, Decoder;
	logic [15:0] SR1_Data, SR2_Data;
	logic [2:0] SR2;
	
	always_comb begin
		unique case (Tri)
			4'b1000	:	BusData = MDR;
			4'b0100	:	BusData = ALU;
			4'b0010	:	BusData = PC;
			4'b0001	:	BusData = Decoder;
			default	: 	BusData = 16'hX;
		endcase
	end
	
	
	ALU alu (.*);
	PC_ pc (.*);
	MAR_ mar (.*);
	MDR_ mdr (.*);
	IR_ ir (.*);
	Decoder decoder (.*);
	reg_file regfile(.*);
	BEN_ ben(.*);
	LED_Display led (.*);

	
endmodule
