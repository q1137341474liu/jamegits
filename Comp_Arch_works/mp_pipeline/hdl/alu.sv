module alu
import rv32i_types::*;
(
    input   logic   [2:0]   aluop,
    input   logic   [31:0]  a, b,
    output  logic   [31:0]  aluout
);

    logic signed   [31:0] as;
    logic signed   [31:0] bs;
    logic unsigned [31:0] au;
    logic unsigned [31:0] bu;

    assign as =   signed'(a);
    assign bs =   signed'(b);
    assign au = unsigned'(a);
    assign bu = unsigned'(b);

    always_comb begin
        unique case (aluop)
            alu_op_add: aluout = au +   bu;
            alu_op_sll: aluout = au <<  bu[3:0];
            alu_op_sra: aluout = unsigned'(as >>> bu[3:0]);
            alu_op_sub: aluout = au -   bu;
            alu_op_xor: aluout = au ^   bu;
            alu_op_srl: aluout = au >>  bu[3:0];
            alu_op_or : aluout = au |   bu;
            alu_op_and: aluout = au &   bu;
            default: aluout = 'x;
        endcase
    end

endmodule: alu
