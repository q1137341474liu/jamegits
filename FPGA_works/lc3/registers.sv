module reg16(input logic[15:0]	Din,
				  input logic 	Clk, Reset, Load,
				  output logic[15:0]	Dout);
				
						always_ff @ (posedge Clk )
				
							begin
								if(Reset)
									Dout <= 16'h0000;
								else if (Load)
									Dout <= Din;
							end
	
endmodule

module reg1(input logic	Din,
				  input logic 	Clk, Reset, Load,
				  output logic	Dout);
				
						always_ff @ (posedge Clk )
				
							begin
								if(Reset)
									Dout <= 1'b0;
								else if (Load)
									Dout <= Din;
							end
	
endmodule


module PC_ (	input Clk, LD_PC, Reset,
						input [1:0] PCMUX,
						input [15:0] BusData, Decoder,
						output logic [15:0] PC);
						
	logic [15:0] Din;
						
	reg16 PC_Reg (.*, .Load(LD_PC), .Dout(PC));
	
	always_comb begin
		unique case (PCMUX)
			2'b00		:	Din = PC + 1;
			2'b01		:	Din = Decoder;
			2'b10		:	Din = BusData;
			default	:	Din = PC;
		endcase
	end
endmodule
			
			

module MAR_ (	input Clk, LD_MAR, Reset,
					input [15:0] BusData,
					output [15:0] MAR);
	
	reg16 MA_Reg	(.*, .Load(LD_MAR), .Din(BusData), .Dout(MAR));
	
endmodule



module MDR_ (	input Clk, LD_MDR, Reset,
					input MIO_EN,
					input [15:0] BusData, MDR_In,
					output logic [15:0] MDR);
					
	logic [15:0] Din;
	
	reg16 MD_Reg (.*, .Load(LD_MDR), .Dout(MDR));
	
	always_comb begin
		if (MIO_EN)
			Din = MDR_In;
		else
			Din = BusData;
	end
endmodule		



module IR_ (	input Clk, LD_IR, Reset,
				input [15:0] BusData,
				output [15:0] IR);
	
	reg16 Instruction_Reg	(.*, .Load(LD_IR), .Din(BusData), .Dout(IR));
	
endmodule


module LED_Display	( 	input LD_LED,
							input [15:0] IR,
							output logic [9:0] LED);
	
	always_comb begin
		if (LD_LED)
			LED = IR[9:0];
		else
			LED = 10'h000;
	end
endmodule

module Decoder (	input	[15:0] IR, PC, SR1_Data,
						input [1:0] ADDR2MUX,
						input ADDR1MUX,
						output logic [15:0] Decoder);

	logic [15:0] Adder1, Adder2;
	mux4 ADDR2_MUX(.A(16'b0),.B({{10{IR[5]}}, IR[5:0]}), .C({{7{IR[8]}}, IR[8:0]}), .D({{5{IR[10]}},IR[10:0]}), .Select(ADDR2MUX), .out(Adder1));
	mux2 ADDR1_MUX(.A(PC), .B(SR1_Data), .Select(ADDR1MUX), .out(Adder2));
	
	always_comb begin
		
		Decoder = Adder1 + Adder2;
	end
endmodule

module reg_file (	input Clk, Reset, LD_REG,
						input DRMUX, SR1MUX,
						input [15:0] IR, BusData,
						output logic [15:0] SR1_Data, SR2_Data);
	logic [7:0] Load;
	logic [15:0] Reg_Out [7:0];
	
	reg16 Reg[7:0] (.*, .Din(BusData), .Dout(Reg_Out));
	
	logic [2:0] SR1, SR2, DR;
	assign SR2 = IR[2:0];
	
	always_comb begin
		if (SR1MUX)
			SR1 = IR[8:6];
		else
			SR1 = IR[11:9];
		
		if (DRMUX)
			DR = 3'b111;
		else
			DR = IR[11:9];
			
		unique case (SR1)
			3'b000	: 	SR1_Data = Reg_Out[0];
			3'b001	: 	SR1_Data = Reg_Out[1];
			3'b010	: 	SR1_Data = Reg_Out[2];
			3'b011	: 	SR1_Data = Reg_Out[3];
			3'b100	: 	SR1_Data = Reg_Out[4];
			3'b101	: 	SR1_Data = Reg_Out[5];
			3'b110	: 	SR1_Data = Reg_Out[6];
			3'b111	: 	SR1_Data = Reg_Out[7];
		endcase
		
		unique case (SR2)
			3'b000	: 	SR2_Data = Reg_Out[0];
			3'b001	: 	SR2_Data = Reg_Out[1];
			3'b010	: 	SR2_Data = Reg_Out[2];
			3'b011	: 	SR2_Data = Reg_Out[3];
			3'b100	: 	SR2_Data = Reg_Out[4];
			3'b101	: 	SR2_Data = Reg_Out[5];
			3'b110	: 	SR2_Data = Reg_Out[6];
			3'b111	: 	SR2_Data = Reg_Out[7];
		endcase
		
		if (LD_REG) 
			begin
				unique case (DR)
					3'b000	: 	Load = 8'h01;
					3'b001	: 	Load = 8'h02;
					3'b010	: 	Load = 8'h04;
					3'b011	: 	Load = 8'h08;
					3'b100	: 	Load = 8'h10;
					3'b101	: 	Load = 8'h20;
					3'b110	: 	Load = 8'h40;
					3'b111	: 	Load = 8'h80;		
				endcase
			end
		else	
			Load = 8'h00;
	end
endmodule

module BEN_(input[15:0] BusData, IR,
			  input Clk, Reset, LD_CC, LD_BEN,
			  output logic BEN);
			  
	logic [2:0] nzp, nzp_out;
	logic	BEN_in;
	
	reg1	NZP_ff[2:0] (.*, .Load(LD_CC), .Din(nzp), .Dout(nzp_out));
	reg1 BEN_ff (.*, .Load(LD_BEN), .Din(BEN_in), .Dout(BEN));
	
	always_comb begin
		if (BusData[15] == 1'b1)
			nzp = 3'b100;
		else if (BusData == 16'h0000)
			nzp = 3'b010;
		else
			nzp = 3'b001;
			
		BEN_in = (nzp_out[0]&IR[9])|(nzp_out[1]&IR[10])|(nzp_out[2]&IR[11]);
		
	end
endmodule
