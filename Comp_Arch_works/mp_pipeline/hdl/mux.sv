module mux (
    	input logic [31:0] a,
	input logic [31:0] b,
    	input logic select,
    	output logic [31:0] c
);
    assign c = (~select) ? a : b ;
    
endmodule

module mux3 (
    	input logic [31:0] a,
	input logic [31:0] b,
	input logic [31:0] c,
	input logic [31:0] d,
    	input logic [1:0] select,
    	output logic [31:0] out
);
    always_comb begin
        case (select)
            2'b00: out = a; // Select first input
            2'b01: out = b; // Select second input
            2'b10: out = c; // Select third input
            2'b11: out = d; // Select fourth input
            default: out = 1'b0;   // Default case (safety)
        endcase
    end



endmodule

