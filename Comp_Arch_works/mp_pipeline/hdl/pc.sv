module pc (

    input logic clk,
    input logic rst,
    input logic [31:0] pc_next,
	//input logic stall,
    output logic [31:0] pc
);
    logic [31:0] pc_reg;

    always_ff @(posedge clk)
    begin
        if(rst == 1'b1)
            pc_reg <= 32'h1eceb000;
        else begin
            pc_reg <= pc_next;
    	end
end


assign pc = pc_reg;

endmodule
