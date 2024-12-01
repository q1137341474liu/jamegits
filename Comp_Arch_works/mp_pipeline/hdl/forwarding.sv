module Forwarding
import rv32i_types::*;    
(
    input logic [4:0]                   id_rs1_s,
    input logic [4:0]                   id_rs2_s,
    input id_ex_stage_reg_t             id_ex_stage_reg,
    input ex_mem_stage_reg_t            ex_mem_stage_reg,
    input mem_wb_stage_reg_t            mem_wb_stage_reg,

    output logic                        forwarding_stall,
    output rs1_forward_id_t         rs1_forward_id,
    output rs2_forward_id_t         rs2_forward_id,
    output rs1_forward_ex_t         rs1_forward_ex,
    output rs2_forward_ex_t         rs2_forward_ex
);

    always_comb begin
        forwarding_stall = '0;
	if (id_ex_stage_reg.rd_s != '0 && id_ex_stage_reg.control_wb.regf_we && id_ex_stage_reg.control_mem.memread && 
            (id_ex_stage_reg.rd_s == id_rs1_s || id_ex_stage_reg.rd_s == id_rs2_s)) begin 
            forwarding_stall = '1;
        end     
    end

always_comb begin
	rs1_forward_id = rs1_s_id_id;
        rs2_forward_id = rs2_s_id_id;

        if (mem_wb_stage_reg.rd_s != '0 && mem_wb_stage_reg.control_wb.regf_we && mem_wb_stage_reg.rd_s == id_rs1_s) begin 
            rs1_forward_id = rs1_s_wb_id;
        end 
        if (mem_wb_stage_reg.rd_s != '0 && mem_wb_stage_reg.control_wb.regf_we && mem_wb_stage_reg.rd_s == id_rs2_s) begin 
            rs2_forward_id = rs2_s_wb_id;
        end 
end

always_comb begin
        rs1_forward_ex = rs1_s_ex_ex;
        if (ex_mem_stage_reg.rd_s != '0 && ex_mem_stage_reg.control_wb.regf_we && ex_mem_stage_reg.rd_s == id_ex_stage_reg.rs1_s) begin 
            rs1_forward_ex = rs1_s_mem_ex;
        end
        else if (mem_wb_stage_reg.rd_s != '0 && mem_wb_stage_reg.control_wb.regf_we && mem_wb_stage_reg.rd_s == id_ex_stage_reg.rs1_s) begin 
            rs1_forward_ex = rs1_s_wb_ex;
        end  

end

always_comb begin
        rs2_forward_ex = rs2_s_ex_ex;
        if (ex_mem_stage_reg.rd_s != '0 && ex_mem_stage_reg.control_wb.regf_we && ex_mem_stage_reg.rd_s == id_ex_stage_reg.rs2_s) begin 
            rs2_forward_ex = rs2_s_mem_ex;
        end else if (mem_wb_stage_reg.rd_s != '0 && mem_wb_stage_reg.control_wb.regf_we && mem_wb_stage_reg.rd_s == id_ex_stage_reg.rs2_s) begin 
            rs2_forward_ex = rs2_s_wb_ex;
        end 
end
endmodule
