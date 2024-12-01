module loop(
    input   logic           clk,
    input   logic           rst,
    output  logic           ack
);

            logic           req;
            logic   [3:0]   req_key;
	    logic           ack_foo;
	    logic 	    ack_bar;

    foo foo(.*);

    bar bar(.*);

	assign ack = ack_bar;

endmodule
