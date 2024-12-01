module ifid (
	input logic clk,
	input logic rst,
	input logic [31:0] pc,
	input logic [31:0] pc_4,
	input logic [31:0] instr_f,
	output logic [31:0] pc_d,
	output logic [31:0] pc_4_d,
	input stall,
	output logic [31:0] instr_d
	
);

logic [31:0] instr_reg;
logic [31:0] pc_before,pc_after,pc_4_before,pc_4_after,pc_stall,pc_4_stall;

    always_ff @(posedge clk) begin
        if (rst == 1'b1) begin
            instr_reg <= '0;
        end
        else if (stall == 1'b0) begin
            instr_reg <= instr_f;
        end else 
            instr_reg <= instr_reg;
    end

    always_comb begin
        if (rst) begin
            instr_d = '0;
            
        end
        else if(stall == 1'b1) begin
            instr_d = instr_reg;
        
        end else begin
            instr_d = instr_f;
            
        end
    end

    always_ff @(posedge clk) begin
	if(rst) begin
	pc_after <= 32'b0;
	pc_4_after <= 32'b0;
	end else begin
	pc_after <= pc;
	pc_4_after <= pc_4;
	end
end

    always_comb begin
        pc_stall = pc_after;
        pc_4_stall= pc_4_after;
        pc_d  = pc_after;
        pc_4_d = pc_4_after;

        if(stall == 1'b1) begin
            pc_before = pc_stall;
            pc_4_before = pc_4_stall;  
            pc_stall = pc_after;
            pc_4_stall = pc_4_after;
        end else begin      
            pc_before = pc;
            pc_4_before = pc_4;          
            pc_d  = pc_after;
            pc_4_d = pc_4_after;      
        end
    end

endmodule
