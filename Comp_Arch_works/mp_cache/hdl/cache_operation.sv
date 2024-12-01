module cache_ops (
    input logic clk,
    input logic rst,
    input logic ufp_read,ufp_write,
    input logic ufp_read_curr,ufp_write_curr,
    input logic dfp_resp,
    input logic [1:0] hit_miss,
    output logic dfp_read,dfp_write,
    output logic ufp_resp,
    output logic [1:0] sram_ops,
    //output logic write_control,
    output logic [2:0] curr_state,
    output logic stall
);

//logic [1:0] curr_state;
logic [2:0] next_state;
//state: 000 -> initial, 001 -> tag compare, 010 -> dirty miss and write back, 011 -> write allocation for miss, 100 -> one more stall after allocation

always_ff @(posedge clk) begin
    if(rst) begin
        curr_state <= '0;
    end else begin
        curr_state <= next_state;
    end
end

always_comb begin
    next_state = curr_state;
    if(curr_state == 3'b000) begin
        if(ufp_read == 1'b1 | ufp_write == 1'b1) begin
            next_state = 3'b001;
        end
        else begin 
            next_state = 3'b000;
        end
    end
    if(curr_state == 3'b001) begin
        if(hit_miss == 2'b00) begin
            if(ufp_read == 1'b1) begin
                next_state = 3'b001;
            end
            else if(ufp_write == 1'b1)begin
                next_state = 3'b001;
            end
            else begin
                next_state = 3'b000;
            end
        end
        if (hit_miss == 2'b01) begin
            next_state = 3'b010;
        end
        if(hit_miss == 2'b10) begin
            next_state = 3'b011;
        end
    end
    if(curr_state == 3'b010) begin
        if(dfp_resp == 1'b1) begin
            next_state = 3'b011;
        end
    end
    if(curr_state == 3'b011) begin
        if(dfp_resp == 1'b1) begin
            next_state = 3'b100;
        end
    end
    if(curr_state == 3'b100) begin
        next_state = 3'b001;
    end
end

always_comb begin
    ufp_resp = 1'b0;
    //write_control = 1'b1;
    sram_ops = 2'b11;
    dfp_read = 1'b0;
    dfp_write = 1'b0;
    stall = 1'b0;
    if(curr_state == 3'b001) begin
        if(hit_miss == 2'b00) begin
            if(ufp_read_curr == 1'b1) begin
                ufp_resp = 1'b1;
                sram_ops = 2'b00; //hit and we read the clean part
            end

            else if ((ufp_write_curr == 1'b1) ) begin
                ufp_resp = 1'b1;
                sram_ops = 2'b01; //hit but we write the dirty part
                //write_control = 1'b0;
            end


        end
        else if(hit_miss == 2'b01) begin
            dfp_write = 1'b1;
        end
        else if (hit_miss == 2'b10) begin
            dfp_read = 1'b1;
        end
    end
    if(curr_state == 3'b010) begin
        dfp_write = 1'b1;
        stall = 1'b1;
    end
    if(curr_state == 3'b011) begin
        dfp_read = 1'b1;
        stall = 1'b1;
        if(dfp_resp == 1'b1) begin
            sram_ops = 2'b10; //miss and we proceed replace
        end
    end
    if(curr_state == 3'b100) begin
        stall = 1'b1;
    end
end




endmodule
