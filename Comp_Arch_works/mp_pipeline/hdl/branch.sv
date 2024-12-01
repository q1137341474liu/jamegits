module branch
import rv32i_types::*;
(
    input   logic   [2:0]   cmpop,
    input   logic   [31:0]  a, b,
    output  logic           br_en
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
        unique case (cmpop)
            branch_f3_beq : br_en = (au == bu);
            branch_f3_bne : br_en = (au != bu);
            branch_f3_blt : br_en = (as <  bs);
            branch_f3_bge : br_en = (as >  bs) || (as == bs);
            branch_f3_bltu: br_en = (au <  bu);
            branch_f3_bgeu: br_en = (au >  bu) || (au == bu);
            default: br_en = 1'bx;
        endcase
    end

endmodule: branch
