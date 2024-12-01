module data_memory (

   	//input logic clk,
	//input logic rst,
	input logic memwrite_m,
	input logic dmem_resp,
	input logic [31:0] rs2_v_m,
	input logic [31:0] dmem_rdata,
	input logic unsign_m,
	input logic [31:0] instr_m,
	input logic [31:0] alu_result_m,
	input logic memread_m,
	output logic [31:0] rdata_m,
	output logic [31:0] dmem_addr,
	output logic [3:0] dmem_rmask,
	output logic [3:0] dmem_wmask,
    	output  logic   [31:0]  dmem_wdata
	
);

logic [3:0] rmask;
logic [3:0] wmask;
mask data_mask(
	.alu_result_m(instr_m),
	.rmask(rmask),
	.wmask(wmask)
);

assign dmem_rmask = rmask;
assign dmem_wmask = wmask;

/*
    always_ff @ (posedge clk) begin
	if(rst == 1) begin
	    	data_out <= 32'b0;
		dmem_addr <= 32'b0;
		dmem_wdata <= 32'b0;
	end else if ((dmem_resp == 1) & (rmask == 4'b1111)) begin
		data_out <= dmem_rdata;
		dmem_addr <= rd1_m + {{20{1'b0}}, instr_m[31:20]};
	end else if ((dmem_resp == 1) & (rmask == 4'b0011) & (unsign_m == 1)) begin
		data_out <= {{16{1'b0}} , dmem_rdata[15:0]};
		dmem_addr <= rd1_m + {{20{1'b0}}, instr_m[31:20]};
	end else if ((dmem_resp == 1) & (rmask == 4'b0011) & (unsign_m == 0)) begin
		data_out <= {{16{dmem_rdata[15]}} , dmem_rdata[15:0]};
		dmem_addr <= rd1_m + {{20{1'b0}}, instr_m[31:20]};
	end else if ((dmem_resp == 1) & (rmask == 4'b0001) & (unsign_m == 1)) begin
		data_out <= {{24{1'b0}} , dmem_rdata[7:0]};
		dmem_addr <= rd1_m + {{20{1'b0}}, instr_m[31:20]};
	end else if ((dmem_resp == 1) & (rmask == 4'b0001) & (unsign_m == 1)) begin
		data_out <= {{24{dmem_rdata[7]}} , dmem_rdata[7:0]};
		dmem_addr <= rd1_m + {{20{1'b0}}, instr_m[31:20]};
	end 

	else if ((w_enable == 1) & (wmask == 4'b1111)) begin
		dmem_wdata <= wdata;
		dmem_addr <= rd1_m + {{20{1'b0}}, instr_m[31:25],instr_m[11:7]};
	end else if ((w_enable == 1) & (wmask == 4'b0011)) begin
		dmem_wdata <= {{16{wdata[15]}} , wdata[15:0]};
		dmem_addr <= rd1_m + {{20{1'b0}}, instr_m[31:25],instr_m[11:7]};
	end else if ((w_enable == 1) & (wmask == 4'b0001)) begin
		dmem_wdata <= {{24{wdata[7]}} , wdata[7:0]};
		dmem_addr <= rd1_m + {{20{1'b0}}, instr_m[31:25],instr_m[11:7]};
	end else begin
		dmem_addr <= dmem_addr;
		dmem_wdata <= dmem_wdata;
	end
    end
*/




//write_address
always_comb begin
	if ((memwrite_m == 1) & (wmask == 4'b1111)) begin
		dmem_wdata = rs2_v_m;
	end else if ((memwrite_m == 1) & (wmask == 4'b0011)) begin
		dmem_wdata = {{16{rs2_v_m[15]}} , rs2_v_m[15:0]};
	end else if ((memwrite_m == 1) & (wmask == 4'b0001)) begin
		dmem_wdata = {{24{rs2_v_m[7]}} , rs2_v_m[7:0]};
	end else begin
		dmem_wdata = '0;		
	end

end

//read_address
always_comb begin
	if ((memread_m == 1) | (memwrite_m == 1)) begin
	dmem_addr = alu_result_m;
	end else begin
	dmem_addr = '0;
	end

end

//read_data

logic [31:0] rdata_reg;
/*
always_ff @ (posedge clk) begin
	if(rst == 1) begin
	    	rdata_reg <= 32'b0;
	end else if ((rmask == 4'b1111) & (memread_m == 1)) begin
		rdata_reg <= dmem_rdata;
	end else if ((rmask == 4'b0011) & (unsign_m == 1)& (memread_m == 1)) begin
		rdata_reg <= {{16{1'b0}} , dmem_rdata[15:0]};
	end else if ((rmask == 4'b0011) & (unsign_m == 0)& (memread_m == 1)) begin
		rdata_reg <= {{16{dmem_rdata[15]}} , dmem_rdata[15:0]};
	end else if ((rmask == 4'b0001) & (unsign_m == 1)& (memread_m == 1)) begin
		rdata_reg <= {{24{1'b0}} , dmem_rdata[7:0]};
	end else if ((rmask == 4'b0001) & (unsign_m == 1)& (memread_m == 1)) begin
		rdata_reg <= {{24{dmem_rdata[7]}} , dmem_rdata[7:0]};
	end else begin
		rdata_reg <= rdata_reg;
	end
end
*/
logic unsign;
assign unsign = unsign_m;
always_comb begin
	if (dmem_resp == 1 & dmem_rdata != '0) begin
		rdata_reg = dmem_rdata;
	end else begin
		rdata_reg = '0;
	end
/*
		if ((rmask == 4'b1111) & (memread_m == 1)) begin
		rdata_reg = dmem_rdata;
		end else if ((rmask == 4'b0011) & (unsign_m == 1)& (memread_m == 1)) begin
		rdata_reg = {{16{1'b0}} , dmem_rdata[15:0]};
		end else if ((rmask == 4'b0011) & (unsign_m == 0)& (memread_m == 1)) begin
		rdata_reg = {{16{dmem_rdata[15]}} , dmem_rdata[15:0]};
		end else if ((rmask == 4'b0001) & (unsign_m == 1)& (memread_m == 1)) begin
		rdata_reg = {{24{1'b0}} , dmem_rdata[7:0]};
		end else if ((rmask == 4'b0001) & (unsign_m == 1)& (memread_m == 1)) begin
		rdata_reg = {{24{dmem_rdata[7]}} , dmem_rdata[7:0]};
		end else begin
		rdata_reg = '0;
		end
	end 
*/

end


always_comb begin
	if(dmem_resp != 0) begin
		rdata_m= rdata_reg;
	end else begin
		rdata_m = '0;
	end
end



endmodule


module mask (
	//input logic clk,
	input logic [31:0] alu_result_m,
	output logic [3:0] rmask,
	output logic [3:0] wmask

);

logic [14:12] funct3;
logic [6:0] opcode;

assign funct3 = alu_result_m[14:12];
assign opcode = alu_result_m[6:0];

always_comb begin
	if ((opcode == 7'b0000011) & (funct3 == 3'b010)) begin
		rmask = 4'b1111;
		wmask = 4'b0000;
	end else if((opcode == 7'b0000011) & ((funct3 == 3'b000)| (funct3 == 3'b100))) begin
		rmask = 4'b0001;
		wmask = 4'b0000;
	end else if ((opcode == 7'b0000011) & ((funct3 == 3'b001)| (funct3 == 3'b101))) begin
		rmask = 4'b0011;
		wmask = 4'b0000;
	end else if ((opcode == 7'b0100011) & (funct3 == 3'b000)) begin
		rmask = 4'b0000;
		wmask = 4'b0001;
	end else if ((opcode == 7'b0100011) & (funct3 == 3'b001)) begin
		rmask = 4'b0000;
		wmask = 4'b0011;
	end else if ((opcode == 7'b0100011) & (funct3 == 3'b010)) begin
		rmask = 4'b0000;
		wmask = 4'b1111;
	end else begin
		rmask = 4'b0000;
		wmask = 4'b0000;		
	end
end


endmodule
