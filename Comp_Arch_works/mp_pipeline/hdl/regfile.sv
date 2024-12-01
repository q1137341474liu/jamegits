module regfile
import rv32i_types::*;
(
    input   logic           clk,
    input   logic           rst,
    input   logic           regf_we,
    input   logic   [31:0]  rd_v,
    input   logic   [4:0]   rs1_s, rs2_s, rd_s,
    output  logic   [31:0]  rs1_v, rs2_v,
    input   rs1_forward_id_t         rs1_forward_id,
    input   rs2_forward_id_t         rs2_forward_id
);

            logic   [31:0]  data [32];

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < 32; i++) begin
                data[i] <= '0;
            end
        end else if (regf_we && (rd_s != 5'd0)) begin
            data[rd_s] <= rd_v;
        end
    end
/*
    always_ff @(posedge clk) begin
        if (rst) begin
            rs1_v <= '0;
            rs2_v <= '0;
        end else begin
            rs1_v <= (rs1_s != 5'd0) ? data[rs1_s] : '0;
            rs2_v <= (rs2_s != 5'd0) ? data[rs2_s] : '0;
        end
    end
*/

    always_comb begin
        if (rst) begin
            rs1_v = 'x;
            rs2_v = 'x;
        end else begin
		case(rs1_forward_id)
                rs1_s_id_id: rs1_v = (rs1_s != 5'd0) ? data[rs1_s] : '0;
                rs1_s_wb_id: rs1_v = (rs1_s != 5'd0) ? rd_v : '0;
            	endcase
		case(rs2_forward_id)
                rs2_s_id_id: rs2_v = (rs2_s != 5'd0) ? data[rs2_s] : '0;
                rs2_s_wb_id: rs2_v = (rs2_s != 5'd0) ? rd_v : '0;
            	endcase
        end
    end


endmodule
