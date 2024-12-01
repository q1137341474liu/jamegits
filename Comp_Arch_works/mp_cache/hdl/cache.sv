module cache (
    input   logic           clk,
    input   logic           rst,

    // cpu side signals, ufp -> upward facing port
    input   logic   [31:0]  ufp_addr,
    input   logic   [3:0]   ufp_rmask,
    input   logic   [3:0]   ufp_wmask,
    output  logic   [31:0]  ufp_rdata,
    input   logic   [31:0]  ufp_wdata,
    output  logic           ufp_resp,

    // memory side signals, dfp -> downward facing port
    output  logic   [31:0]  dfp_addr,
    output  logic           dfp_read,
    output  logic           dfp_write,
    input   logic   [255:0] dfp_rdata,
    output  logic   [255:0] dfp_wdata,
    input   logic           dfp_resp
);
logic ufp_read,ufp_write;
logic ufp_read_curr, ufp_write_curr;
logic [1:0] read_wayin, read_wayout;
logic [1:0] write_wayin, write_wayout;

//data in 4 different ways 
logic [255:0] cache_rdata[4];
logic [255:0] cache_wdata[4];
logic [31:0] cache_wmask[4];
logic cache_data_web0[4];

//tag in 4 different ways
logic [23:0] cache_wtag[4];
logic [23:0] cache_rtag[4];
logic cache_tag_web0[4];

//valid bits in 4 different ways
logic cache_wvalid[4];
logic cache_rvalid[4];
logic cache_valid_web[4];

logic csb0;
logic [1:0] hit_miss;
logic write_control;

logic [2:0] curr_state;
logic [1:0]way_kick;
logic [1:0]sram_ops;


//pipeline for ufp
logic [3:0]ufp_rmask_curr,ufp_rmask_next,ufp_wmask_curr,ufp_wmask_next;
logic [31:0] ufp_addr_curr,ufp_addr_next;
logic [31:0] ufp_wdata_curr,ufp_wdata_next;
logic stall;
always_ff @(posedge clk) begin
    if(rst) begin
        ufp_rmask_curr <= '0;
        ufp_wmask_curr <= '0;
        ufp_addr_curr <= '0;
        ufp_wdata_curr <= '0;
    end
    else begin
        ufp_rmask_curr <= ufp_rmask_next;
        ufp_wmask_curr <= ufp_wmask_next;
        ufp_addr_curr <= ufp_addr_next;
        ufp_wdata_curr <= ufp_wdata_next;
    end 
end 

//pipeline for stall  -- simply 2:1 mux with stall select
always_comb begin
    if(stall) begin
        ufp_rmask_next = ufp_rmask_curr;
        ufp_wmask_next = ufp_wmask_curr;
        ufp_addr_next = ufp_addr_curr;
        ufp_wdata_next = ufp_wdata_curr;
    end
    else begin
        ufp_rmask_next = ufp_rmask;
        ufp_wmask_next = ufp_wmask;
        ufp_addr_next = ufp_addr;
        ufp_wdata_next = ufp_wdata;    
    end
end

//second mux: using write control to select between next and curr
//output goes into sram and do the tag comparison parts and so on
logic [3:0]cache_rmask_input,cache_rmask_output,cache_wmask_input,cache_wmask_output;
logic [31:0] cache_addr_input,cache_addr_output;
logic [31:0] cache_wdata_input,cache_wdata_output;
assign write_control = 1'b0;
always_comb begin
    if(write_control) begin
        cache_rmask_input = ufp_rmask_curr;
        cache_wmask_input = ufp_wmask_curr;
        cache_addr_input = ufp_addr_curr;
        cache_wdata_input = ufp_wdata_curr; 
    end
    else begin
        cache_rmask_input = ufp_rmask_next;
        cache_wmask_input = ufp_wmask_next;
        cache_addr_input = ufp_addr_next;
        cache_wdata_input = ufp_wdata_next; 
    end
end  

always_ff @(posedge clk) begin
	if(rst) begin
	cache_wdata_output <= '0;
	cache_addr_output <= '0;
	cache_rmask_output <= '0;
	cache_wmask_output <= '0;
	end
	else begin
	cache_wdata_output <= cache_wdata_input;
	cache_addr_output <= cache_addr_input;
	cache_rmask_output <= cache_rmask_input;
	cache_wmask_output <= cache_wmask_input;	
	end
end    

//set tag and set from address bits mapping
logic [3:0] set;
assign set = ufp_addr_curr[8:5];
logic [22:0] tag;
assign tag = ufp_addr_curr[31:9];

//tag_comparing: if miss or hit
always_comb begin
    read_wayout = 'x;
    hit_miss = 'x;
    if(curr_state == 3'b001) begin
        if((cache_rtag[0][22:0] == tag) && (cache_rvalid[0] == 1'b1)) begin
            read_wayout = 2'b00; //way A
            hit_miss = 2'b00;
        end
        else if((cache_rtag[1][22:0] == tag) && (cache_rvalid[1] == 1'b1)) begin
            read_wayout = 2'b01; //way B
            hit_miss = 2'b00;
        end
        else if((cache_rtag[2][22:0] == tag) && (cache_rvalid[2] == 1'b1)) begin
            read_wayout = 2'b10; //way C
            hit_miss = 2'b00;
        end
        else if((cache_rtag[3][22:0] == tag) && (cache_rvalid[3] == 1'b1)) begin
            read_wayout = 2'b11; //way D
            hit_miss = 2'b00;
        end
        else if((cache_rtag[way_kick][23] == 1'b1) && (cache_rvalid[way_kick] == 1'b1)) begin
            hit_miss = 2'b01; //dirty miss
        end
        else begin
            hit_miss = 2'b10; //clean miss
        end
    end
	else begin
	hit_miss = 2'b11;
	end
end

    always_comb begin
        ufp_rdata = 'x;
        if (sram_ops == 2'b00) begin
            ufp_rdata = cache_rdata[read_wayout][32 * ufp_addr_curr[4:2] +: 32];
        end 
    end 

//determine when ufp_read/ufp_write
always_comb begin
    ufp_write = 1'b0;
    ufp_read = 1'b0;
    //if((cache_wmask_input != '0) & (cache_wdata_input == ufp_wdata_next)) begin
	if(cache_wmask_input != '0) begin
        ufp_write = 1'b1;
    end
    if(cache_rmask_input != '0) begin
        ufp_read = 1'b1;
    end
end
//current value for read/write
always_comb begin
    ufp_write_curr = 1'b0;
    ufp_read_curr = 1'b0;
    //if((cache_wmask_input != '0) & (cache_wdata_input == ufp_wdata_next)) begin
	if(cache_wmask_output != '0) begin
        ufp_write_curr = 1'b1;
    end
    if(cache_rmask_output != '0) begin
        ufp_read_curr = 1'b1;
    end
end

//csb0 for cache: when addr != 0, csb0 = 0
always_comb begin
    csb0 = 1'b1;
    if(cache_addr_input != '0) begin
        csb0 = 1'b0;
    end
end

//stall logic
/*
always_comb begin
    stall = 1'b0;
    if((hit_miss == 2'b01) | (hit_miss == 2'b10)) begin
        stall = 1'b1;
    end
    else if ((cache_wdata_output != ufp_wdata_curr) & (ufp_wmask_curr != '0) & (cache_wmask_output != '0)) begin
        stall = 1'b1;
    end
    else if ((ufp_rmask_curr != '0) & (cache_wmask_output != 0)) begin
        stall = 1'b1;
    end
end
*/


//if clean miss or dirty miss
//1st: write enable
always_comb begin 
    for (int i = 0; i < 4; i ++) begin 
        cache_data_web0[i] = 1'b1;
        cache_tag_web0[i] = 1'b1;
        cache_valid_web[i] = 1'b1;
    end 
    if (sram_ops == 2'b01) begin 
        cache_data_web0[read_wayout] = 1'b0;
        cache_tag_web0[read_wayout] = 1'b0;
        cache_valid_web[read_wayout] = 1'b0;
    end 
    else if (sram_ops == 2'b10) begin 
        cache_data_web0[way_kick] = 1'b0;
        cache_tag_web0[way_kick] = 1'b0;
        cache_valid_web[way_kick] = 1'b0;
    end 
end
//2nd: replace way_kick values with rdata, set valid = 1, tag to clean
//2nd：
always_comb begin
    for (int i = 0; i < 4; i ++) begin 
        cache_wdata[i] = '0;
        cache_wtag[i] = '0;
        cache_wvalid[i] = '0;
        cache_wmask[i] = '0;
    end
    if(sram_ops == 2'b10) begin
        cache_wdata[way_kick] = dfp_rdata;
        cache_wtag[way_kick] = {1'b0, tag};
        cache_wvalid[way_kick] = 1'b1;
        cache_wmask[way_kick] = '1;
    end  
    if(sram_ops == 2'b01) begin
        cache_wdata[read_wayout][32 * ufp_addr_curr[4:2] +: 32] = ufp_wdata_curr;
        cache_wtag[read_wayout] = {1'b1, tag};
        cache_wvalid[read_wayout] = 1'b1;
        cache_wmask[read_wayout] = 32'hffffffff;//ufp_wmask_curr << (ufp_addr_curr[4:2]*4);
    end         
end

//write back to memory since we kick out dirty
always_comb begin 
    dfp_addr = '0;
    dfp_wdata = 'x;
    if (dfp_write == 1'b1) begin 
        dfp_addr = {cache_rtag[way_kick][22:0] , set, 5'b0};
        dfp_wdata = cache_rdata[way_kick];
    end else if (dfp_read == 1'b1) begin
        dfp_addr = {tag, set, 5'b0};
    end 
end 

//cache fsm
cache_ops cache_operation (
    .clk(clk),
    .rst(rst),
    .ufp_read(ufp_read),
    .ufp_write(ufp_write),
    .ufp_read_curr(ufp_read_curr),
    .ufp_write_curr(ufp_write_curr),
    .dfp_resp(dfp_resp),
    .ufp_resp(ufp_resp),
    .sram_ops(sram_ops),
    .hit_miss(hit_miss),
    .dfp_read(dfp_read),
    .dfp_write(dfp_write),
    //.write_control(write_control),
    .curr_state(curr_state),
    .stall(stall)
);

    generate for (genvar i = 0; i < 4; i++) begin : arrays
        mp_cache_data_array data_array (
            .clk0       (clk),
            .csb0       (csb0),
            .web0       (cache_data_web0[i]),
            .wmask0     (cache_wmask[i]),
            .addr0      (set),
            .din0       (cache_wdata[i]),
            .dout0      (cache_rdata[i])
        );
        mp_cache_tag_array tag_array (
            .clk0       (clk),
            .csb0       (csb0),
            .web0       (cache_tag_web0[i]),
            .addr0      (set),
            .din0       (cache_wtag[i]),
            .dout0      (cache_rtag[i])
        );
        valid_array valid_array (
            .clk0       (clk),
            .rst0       (rst),
            .csb0       (csb0),
            .web0       (cache_valid_web[i]),
            .addr0      (set),
            .din0       (cache_wvalid[i]),
            .dout0      (cache_rvalid[i])
        );
    end endgenerate

//port A for read, port B for write
//lru: if hit, change the lru_out based on the lru_in and read_wayout; if miss, keep it and pick what way to be kicked
//way_kick only used for miss replace
logic [2:0]lru_in;
logic [2:0]lru_out;
always_comb begin
	lru_out = lru_in;
    if(hit_miss == 2'b00) begin
        if (read_wayout == 2'b00) begin
            lru_out = {1'b0,1'b0,lru_in[0]}; //Way A
        end
        if (read_wayout == 2'b01) begin
            lru_out = {1'b0,1'b1,lru_in[0]}; //Way B
        end
        if (read_wayout == 2'b10) begin
            lru_out = {1'b1,lru_in[1],1'b0}; //Way C
        end
        if (read_wayout == 2'b11) begin
            lru_out = {1'b1,lru_in[1],1'b1}; //Way D
        end 
    end
end


always_comb begin
    way_kick = 'x;
	//if(hit_miss == 2'b10) begin
    if(lru_in[0] == 1'b1) begin
        if(lru_in[1] == 1'b1) begin
            way_kick = 2'b00;
        end
        if(lru_in[1] == 1'b0) begin
            way_kick = 2'b01;
        end
    end
    else if(lru_in[1] == 1'b1) begin
        if(lru_in[2] == 1'b1) begin
            way_kick = 2'b10;
        end
        if(lru_in[2] == 1'b0) begin
            way_kick = 2'b11;
        end
    end
//end
end


logic lru_csb0;
logic lru_csb1;
logic lru_web1;

assign lru_csb0 = 1'b0;
assign lru_csb1 = 1'b0;
assign lru_web1 = 1'b0;


logic [2:0]dout1;
logic [2:0]din0;
assign din0 = {1'b0,1'b0,1'b0};
    lru_array lru_array (
        .clk0       (clk),
        .rst0       (rst),
        .csb0       (lru_csb0),
        .web0       (1'b1),
        .addr0      (set),
        .din0       (din0),
        .dout0      (lru_in),
        .csb1       (lru_csb1),
        .web1       (lru_web1),
        .addr1      (set),
        .din1       (lru_out),
        .dout1      (dout1)
    );

endmodule
