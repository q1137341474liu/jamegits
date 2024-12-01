module fetch
import rv32i_types::*;
(
	input logic clk,
	input logic rst,
	input logic stall,
	input logic flush,
	input logic go,
	input logic [31:0] pc_result,


	output logic [31:0] imem_addr,
	output logic [3:0]  imem_rmask,
	output logic imem_trigger,
	output if_id_stage_reg_t if_id_stage_reg

);      
    logic   [31:0]  pc;
    logic   [31:0]  pc_next;

    logic   [63:0]  order;
    logic   [63:0]  order_next;

//pc source and order counters
    always_ff @( posedge clk ) begin
        if (rst == 1) begin 
            pc <= 32'h1eceb000;
            order <= '0;
        end else begin 
        	if (stall == 0) begin 
                	if (go) begin 
                    	pc <= pc_next;
                    	order <= order_next;
                	end
        	end
        end 
    end

//flush take from branch,clear pc and update new one
    always_comb begin
        if (flush == 1) begin 
            pc_next = pc_result;
            order_next = order - 'd1;
        end else begin 
            pc_next = pc + 'd4 ;
            order_next = order + 'd1;
        end 
    end 

//only give out instruction when our "go" sign is ready
    always_comb begin
        imem_addr = '0;
        imem_rmask = '0;
        imem_trigger = '0;
        if (stall == 0) begin
        	if (go) begin 
                	imem_addr = pc;
                	imem_rmask = 4'b1111;
                	imem_trigger = 1'b1;
            	end else begin
        		imem_addr = '0;
        		imem_rmask = '0;
        		imem_trigger = '0;		
	    	end
        end 
    end


//if_id reg store value
    always_comb begin 
        if_id_stage_reg.pc = pc;
        if_id_stage_reg.pc_next = pc_next;
        if_id_stage_reg.order = order;
        if_id_stage_reg.valid = 1'b1;
        if (flush == 1) begin 
        if_id_stage_reg.pc = '0;
        if_id_stage_reg.pc_next = '0;
        if_id_stage_reg.order = '0;
        if_id_stage_reg.valid = '0;
        end
    end 

endmodule
