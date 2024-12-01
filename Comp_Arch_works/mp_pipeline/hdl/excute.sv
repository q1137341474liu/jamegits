module excute
import rv32i_types::*;
(
	input logic [31:0] rd_v_wb,
	input id_ex_stage_reg_t id_ex_stage_reg,
	input ex_mem_stage_reg_t ex_mem_stage_reg_in,
	input rs1_forward_ex_t rs1_forward_ex,
	input rs2_forward_ex_t rs2_forward_ex,
	output ex_mem_stage_reg_t ex_mem_stage_reg_out,
	output logic [31:0] BrPC,
	output logic flush


);

            logic   [63:0]  order;
            logic   valid;
            logic   [31:0]  pc;
            logic   [31:0]  pc_next;
  

            logic   [31:0]  inst;
            logic   [6:0]   opcode;
assign opcode = inst[6:0];
            logic   [31:0]  i_imm;
            logic   [31:0]  s_imm;
            logic   [31:0]  b_imm;
            logic   [31:0]  u_imm;
            logic   [31:0]  j_imm;
            logic   [31:0]  rs1_v;
            logic   [31:0]  rs2_v;
            logic   [31:0]  rs1_v_ex;
            logic   [31:0]  rs2_v_ex;
            logic   [31:0]  rs1_v_mem;
            logic   [31:0]  rs2_v_mem;
            logic   [31:0]  rs1_v_wb;
            logic   [31:0]  rs2_v_wb;

            logic           regf_we;
            logic   [4:0]  rs1_s;
            logic   [4:0]  rs2_s;
            logic   [4:0]  rd_s;

    control_ex_t     control_ex;
    control_mem_t    control_mem;
    control_wb_t     control_wb;
	logic [31:0] mem_addr;
	logic [3:0] mem_rmask;
	logic [3:0] mem_wmask;
	logic [31:0] mem_wdata;

	logic [31:0] alu_result;
	logic branch_out;

    always_comb begin 
        order = id_ex_stage_reg.order;
        valid = id_ex_stage_reg.valid;
        pc = id_ex_stage_reg.pc;
        inst = id_ex_stage_reg.inst;
        i_imm = id_ex_stage_reg.i_imm;
        s_imm = id_ex_stage_reg.s_imm;
        b_imm = id_ex_stage_reg.b_imm;
        u_imm = id_ex_stage_reg.u_imm;
        j_imm = id_ex_stage_reg.j_imm;


        control_ex = id_ex_stage_reg.control_ex;
        control_mem = id_ex_stage_reg.control_mem;
        control_wb = id_ex_stage_reg.control_wb;

        rs1_v_ex = id_ex_stage_reg.rs1_v;
        rs2_v_ex = id_ex_stage_reg.rs2_v;
        rs1_s = id_ex_stage_reg.rs1_s;
        rs2_s = id_ex_stage_reg.rs2_s;
        rd_s = id_ex_stage_reg.rd_s;
    end 


    always_comb begin
	rs1_v_mem = 'x;
	rs2_v_mem = 'x;
        rs1_v_wb = rd_v_wb;
        rs2_v_wb = rd_v_wb;
	case(ex_mem_stage_reg_in.control_wb.regf_mux)
		alu_out_wb: begin
		rs1_v_mem = ex_mem_stage_reg_in.alu_out;
		rs2_v_mem = ex_mem_stage_reg_in.alu_out;
		end
		branch_wb: begin
		rs1_v_mem = ex_mem_stage_reg_in.branch;
		rs2_v_mem = ex_mem_stage_reg_in.branch;
		end
		u_imm_wb: begin	
                rs1_v_mem = ex_mem_stage_reg_in.u_imm;
                rs2_v_mem = ex_mem_stage_reg_in.u_imm;
                end 	
		pc_4_wb: begin	
                rs1_v_mem = ex_mem_stage_reg_in.pc + 'd4;
                rs2_v_mem = ex_mem_stage_reg_in.pc + 'd4;
                end 
	endcase						
    end

/*

always_ff @(posedge clk) begin
	if(rst) begin
	memtoreg_m <= '0;
	end else begin
	memtoreg_m <= memtoreg_e;
	end
end
*/


//modified version

always_comb begin 
        rs1_v = 'x;
        rs2_v = 'x;
        case (rs1_forward_ex)  
            rs1_s_ex_ex: rs1_v = rs1_v_ex;
            rs1_s_mem_ex: rs1_v = rs1_v_mem;
            rs1_s_wb_ex:  rs1_v = rs1_v_wb;
        endcase 
        case (rs2_forward_ex)  
            rs2_s_ex_ex: rs2_v = rs2_v_ex;
            rs2_s_mem_ex: rs2_v = rs2_v_mem;
            rs2_s_wb_ex:  rs2_v = rs2_v_wb;
        endcase
    end


    // branch
    always_comb begin
        pc_next = id_ex_stage_reg.pc_next;
        flush = '0;
        BrPC = '0;
        case (opcode)
            op_b_jal: begin 
                if (pc_next != ((pc + j_imm))) begin 
                    flush = '1;
                    BrPC = pc + j_imm;
                    pc_next = pc + j_imm;
                end 
            end 
            op_b_jalr: begin 
                if (pc_next != ((rs1_v + i_imm) & 32'hfffffffe)) begin 
                    flush = '1;
                    BrPC = (rs1_v + i_imm) & 32'hfffffffe;
                    pc_next = (rs1_v + i_imm) & 32'hfffffffe;
                end
            end 
            op_b_br: begin
                if (branch_out) begin 
                    if (pc_next != ((pc + b_imm))) begin 
                        flush = '1;
                        BrPC = pc + b_imm;
                        pc_next = pc + b_imm;
                    end
                end
            end 
        endcase
    end 

    // alu_m1
    logic [31:0]    alu_m1_sel_grab;
    always_comb begin 
        alu_m1_sel_grab = 'x;
        case (control_ex.alumux1_sel)
            rs1_out_alu: alu_m1_sel_grab = rs1_v;
            pc_out_alu: alu_m1_sel_grab = pc;
        endcase
    end 

    // alu_m2
    logic [31:0]    alu_m2_sel_grab;
    always_comb begin 
        alu_m2_sel_grab = 'x;
        case (control_ex.alumux2_sel)
            i_imm_alu: alu_m2_sel_grab = i_imm;
            u_imm_alu: alu_m2_sel_grab = u_imm;  
            b_imm_alu: alu_m2_sel_grab = b_imm;  
            s_imm_alu: alu_m2_sel_grab = s_imm;  
            j_imm_alu: alu_m2_sel_grab = j_imm;
            rs2_out_alu: alu_m2_sel_grab = rs2_v;
        endcase
    end 

    // cmp_m
    logic [31:0]   cmp_m_sel_grab;
    always_comb begin
        cmp_m_sel_grab = 'x;
        case (control_ex.branch_mux) 
            rs2_branch_ex: cmp_m_sel_grab = rs2_v;
            i_imm_branch_ex: cmp_m_sel_grab = i_imm;
        endcase
    end 

    // alu
    alu alu(
        .aluop(control_ex.alu_ops),
        .a(alu_m1_sel_grab),
        .b(alu_m2_sel_grab),
        .aluout(alu_result)
    );

    // cmp
    branch branch(
        .cmpop(control_ex.branch_f3),
        .a(rs1_v),
        .b(cmp_m_sel_grab),
        .br_en(branch_out)
    );


    // dmem
    always_comb begin
        mem_addr = '0;
        mem_wdata = '0;
        mem_rmask = '0;
        mem_wmask = '0;

        if (control_mem.memread) begin
            case (control_mem.load_ops) 
                load_lb, load_lbu: begin 
                    mem_addr = alu_result;
                    mem_rmask = 4'b0001 << mem_addr[1:0];
                end
                load_lh, load_lhu: begin 
                    mem_addr = alu_result;
                    mem_rmask = 4'b0011 << mem_addr[1:0];
                end 
                load_lw: begin
                    mem_addr = alu_result;
                    mem_rmask = 4'b1111;
                end 
            endcase
        end else if (control_mem.memwrite) begin 
            case (control_mem.store_ops)
                store_sb: begin 
                    mem_addr = alu_result;
                    mem_wmask = 4'b0001 << mem_addr[1:0];
                    mem_wdata[8 * mem_addr[1:0] +: 8] = rs2_v[7:0];
                end
                store_sh: begin  
                    mem_addr = alu_result;
                    mem_wmask = 4'b0011 << mem_addr[1:0];
                    mem_wdata[16 * mem_addr[1] +: 16] = rs2_v[15:0];
                end
                store_sw: begin
                    mem_addr = alu_result;
                    mem_wmask = 4'b1111;
                    mem_wdata = rs2_v;
                end
            endcase
        end
    end 

    always_comb begin
        ex_mem_stage_reg_out.inst = inst;
        ex_mem_stage_reg_out.pc = pc;
        ex_mem_stage_reg_out.pc_next = pc_next;
        ex_mem_stage_reg_out.order = order;
        ex_mem_stage_reg_out.valid = valid; 
        ex_mem_stage_reg_out.control_mem = control_mem;
        ex_mem_stage_reg_out.control_wb = control_wb;
        ex_mem_stage_reg_out.alu_out = alu_result;
        ex_mem_stage_reg_out.branch = {31'd0, branch_out};
        ex_mem_stage_reg_out.u_imm = u_imm;
        ex_mem_stage_reg_out.rs1_v = rs1_v;
        ex_mem_stage_reg_out.rs2_v = rs2_v;
        ex_mem_stage_reg_out.rs1_s = rs1_s;
        ex_mem_stage_reg_out.rs2_s = rs2_s;
        ex_mem_stage_reg_out.rd_s = rd_s;   
        ex_mem_stage_reg_out.mem_addr = mem_addr;
        ex_mem_stage_reg_out.mem_rmask = mem_rmask;
        ex_mem_stage_reg_out.mem_wmask = mem_wmask;
        ex_mem_stage_reg_out.mem_wdata = mem_wdata;
    end 


endmodule
