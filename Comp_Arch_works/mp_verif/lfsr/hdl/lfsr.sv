module lfsr #(
    parameter bit   [15:0]  SEED_VALUE = 'hECEB
) (
    input   logic           clk,
    input   logic           rst,
    input   logic           en,
    output  logic           rand_bit,
    output  logic   [15:0]  shift_reg
);

    // TODO: Fill this out!

logic [15:0]s;
logic feedback;
assign feedback = ((s[2]^s[0])^s[3])^s[5];


always_ff @ (posedge clk)
begin
if (rst) begin
s <= SEED_VALUE;
end

else if (en) begin
s <= {feedback,s[15:1]};
rand_bit <= s[0];
end

else begin
s <= s;
end

end

assign shift_reg = s;
endmodule
