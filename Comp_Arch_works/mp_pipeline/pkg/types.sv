/////////////////////////////////////////////////////////////
// Maybe merge what is in mp_verif/pkg/types.sv over here? //
/////////////////////////////////////////////////////////////

package rv32i_types;

    typedef enum logic {
        rs1_out_alu = 1'b0,
        pc_out_alu  = 1'b1
    } alumux1_sel_t;
    typedef enum bit [2:0] {
        i_imm_alu    = 3'b000
        ,u_imm_alu   = 3'b001
        ,b_imm_alu   = 3'b010
        ,s_imm_alu   = 3'b011
        ,j_imm_alu   = 3'b100
        ,rs2_out_alu = 3'b101
    } alumux2_sel_t;
    typedef enum bit {
        rs2_branch_ex = 1'b0,
        i_imm_branch_ex = 1'b1
    } branch_mux_t;

    typedef enum bit[1:0] {
        rs1_s_ex_ex = 2'b00,
        rs1_s_mem_ex = 2'b01,
        rs1_s_wb_ex = 2'b10 
    } rs1_forward_ex_t;
    typedef enum bit[1:0] {
        rs2_s_ex_ex = 2'b00,
        rs2_s_mem_ex = 2'b01,
        rs2_s_wb_ex = 2'b10 
    } rs2_forward_ex_t;
    // transparent register
    typedef enum bit {
        rs1_s_id_id = 1'b0,
        rs1_s_wb_id = 1'b1
    } rs1_forward_id_t;
    typedef enum bit {
        rs2_s_id_id = 1'b0,
        rs2_s_wb_id = 1'b1
    } rs2_forward_id_t;
    // more mux def here


    typedef enum bit [6:0] {
        op_b_lui       = 7'b0110111, // load upper immediate (U type)
        op_b_auipc     = 7'b0010111, // add upper immediate PC (U type)
        op_b_jal       = 7'b1101111, // jump and link (J type)
        op_b_jalr      = 7'b1100111, // jump and link register (I type)
        op_b_br        = 7'b1100011, // branch (B type)
        op_b_load      = 7'b0000011, // load (I type)
        op_b_store     = 7'b0100011, // store (S type)
        op_b_imm       = 7'b0010011, // arith ops with register/immediate operands (I type)
        op_b_reg       = 7'b0110011  // arith ops with register operands (R type)
    } rv32i_opcode;

    typedef enum logic [2:0] {
        arith_f3_add   = 3'b000, // check logic 30 for sub if op_reg op
        arith_f3_sll   = 3'b001,
        arith_f3_slt   = 3'b010,
        arith_f3_sltu  = 3'b011,
        arith_f3_xor   = 3'b100,
        arith_f3_sr    = 3'b101, // check logic 30 for logical/arithmetic
        arith_f3_or    = 3'b110,
        arith_f3_and   = 3'b111
    } arith_f3_t;

    typedef enum logic [2:0] {
        load_f3_lb     = 3'b000,
        load_f3_lh     = 3'b001,
        load_f3_lw     = 3'b010,
        load_f3_lbu    = 3'b100,
        load_f3_lhu    = 3'b101
    } load_f3_t;

    typedef enum logic [2:0] {
        store_f3_sb    = 3'b000,
        store_f3_sh    = 3'b001,
        store_f3_sw    = 3'b010
    } store_f3_t;

    typedef enum logic [2:0] {
        branch_f3_beq  = 3'b000,
        branch_f3_bne  = 3'b001,
        branch_f3_blt  = 3'b100,
        branch_f3_bge  = 3'b101,
        branch_f3_bltu = 3'b110,
        branch_f3_bgeu = 3'b111
    } branch_f3_t;

    typedef enum logic [2:0] {
        alu_op_add     = 3'b000,
        alu_op_sll     = 3'b001,
        alu_op_sra     = 3'b010,
        alu_op_sub     = 3'b011,
        alu_op_xor     = 3'b100,
        alu_op_srl     = 3'b101,
        alu_op_or      = 3'b110,
        alu_op_and     = 3'b111
    } alu_ops;

    typedef enum bit [6:0] {
        base_funct7    = 7'b0000000,
        variant_funct7 = 7'b0100000
    } arith_f7_t;

    typedef enum bit [2:0] {
        load_lb  = 3'b000,
        load_lh  = 3'b001,
        load_lw  = 3'b010,
        load_lbu = 3'b100,
        load_lhu = 3'b101
    } load_ops;
    typedef enum bit [2:0] {
        store_sb = 3'b000,
        store_sh = 3'b001,
        store_sw = 3'b010
    } store_ops;

    typedef struct packed {
        logic                   alumux1_sel;
        logic [2:0]             alumux2_sel;
        logic [2:0]             alu_ops;
        logic                   branch_mux;
        logic [2:0]             branch_f3;
    }control_ex_t;
    typedef struct packed {
        logic               memread;
        logic               memwrite;
        logic [2:0]         load_ops;
        logic [2:0]         store_ops;
    }control_mem_t;
    typedef struct packed {
        logic [3:0]         regf_mux;
        logic               regf_we;
    }control_wb_t;

    typedef enum bit [3:0] {
        alu_out_wb   = 4'b0000,
        branch_wb    = 4'b0001,
        u_imm_wb    = 4'b0010,
        lw_wb       = 4'b0011,
        pc_4_wb = 4'b0100,
        lb_wb        = 4'b0101,
        lbu_wb       = 4'b0110,  // unsigned byte
        lh_wb        = 4'b0111,
        lhu_wb       = 4'b1000  // unsigned halfword
    } regf_mux_t;

    typedef struct packed {
        logic   [31:0]      pc;
        logic   [31:0]      pc_next;
        logic   [63:0]      order;
        logic               valid;
    } if_id_stage_reg_t;

    typedef struct packed {
        logic   [31:0]      inst;
        logic   [31:0]      pc;
        logic   [63:0]      order;
	logic	[31:0]	pc_next;
	logic	valid;
        logic   [4:0]       rs1_s;
        logic   [4:0]       rs2_s;
        logic   [31:0]      rs1_v;
        logic   [31:0]      rs2_v;
        logic   [4:0]       rd_s;
        logic   [31:0]      i_imm;
        logic   [31:0]      s_imm;
        logic   [31:0]      b_imm;
        logic   [31:0]      u_imm;
        logic   [31:0]      j_imm;
	control_ex_t control_ex;
	control_mem_t control_mem;
	control_wb_t control_wb;

        // what else?

    } id_ex_stage_reg_t;

    typedef struct packed {
        logic   [31:0]      inst;
        logic   [31:0]      pc;
        logic   [63:0]      order;
	logic	[31:0]	pc_next;
	logic	valid;
        logic   [4:0]       rs1_s;
        logic   [4:0]       rs2_s;
        logic   [31:0]      rs1_v;
        logic   [31:0]      rs2_v;
        logic   [4:0]       rd_s;
        logic   [31:0]      u_imm;

	logic   [31:0]	    branch;
	logic [31:0] alu_out;

	control_mem_t control_mem;
	control_wb_t control_wb;
        logic [31:0] mem_addr;
        logic [3:0] mem_rmask;
        logic [3:0] mem_wmask;
        logic [31:0] mem_wdata;

    } ex_mem_stage_reg_t;

    typedef struct packed {
        logic   [31:0]      inst;
        logic   [31:0]      pc;
        logic   [63:0]      order;
	logic	[31:0]	pc_next;
	logic	valid;
        logic   [4:0]       rs1_s;
        logic   [4:0]       rs2_s;
        logic   [31:0]      rs1_v;
        logic   [31:0]      rs2_v;
        logic   [4:0]       rd_s;
        logic   [31:0]      u_imm;

	logic   [31:0]	    branch;
	logic [31:0] alu_out;

	control_mem_t control_mem;
	control_wb_t control_wb;
        logic [31:0] mem_addr;
        logic [3:0] mem_rmask;
        logic [3:0] mem_wmask;
        logic [31:0] mem_wdata;
	logic [31:0] load_type;
	logic [31:0] mem_rdata;
    } mem_wb_stage_reg_t;

endpackage
