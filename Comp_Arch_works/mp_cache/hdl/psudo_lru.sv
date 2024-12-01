module pseudo_lru(
    input logic clk,
    input logic rst,
    input logic ufp_resp,
    input logic [1:0] way, //[A,B,C,D] = [00,01,10,11]
    input logic [3:0] set, //16 sets -> 16 lru_states seperately
    output logic [1:0] way_kick  // [A,B,C,D]
    
);

logic [2:0] lru_state [16]; //16 sets with 3 bits to indicate current state
logic [2:0] lru_state_next;

always_ff @(posedge clk) begin
    if(rst) begin
        lru_state[0] <= '0;
        lru_state[1] <= '0;
        lru_state[2] <= '0;
        lru_state[3] <= '0;
        lru_state[4] <= '0;
        lru_state[5] <= '0;
        lru_state[6] <= '0;
        lru_state[7] <= '0;
        lru_state[8] <= '0;
        lru_state[9] <= '0;
        lru_state[10] <= '0;
        lru_state[11] <= '0;
        lru_state[12] <= '0;
        lru_state[13] <= '0;
        lru_state[14] <= '0;
        lru_state[15] <= '0;
    end
    else if (ufp_resp == 1'b1) begin
        lru_state[set] <= lru_state_next;
    end
end


always_comb begin
    lru_state_next = lru_state[set];
    if (way == 2'b00) begin
        lru_state_next = {1'b0,1'b0,lru_state[set][0]}; //Way A
    end
    if (way == 2'b01) begin
        lru_state_next = {1'b0,1'b1,lru_state[set][0]}; //Way B
    end
    if (way == 2'b10) begin
        lru_state_next = {1'b1,lru_state[set][1],1'b0}; //Way C
    end
    if (way == 2'b11) begin
        lru_state_next = {1'b1,lru_state[set][1],1'b1}; //Way D
    end    

end

always_comb begin
    way_kick = 'x;
    if(lru_state[set][0] == 1'b1) begin
        if(lru_state[set][1] == 1'b1) begin
            way_kick = 2'b00;
        end
        if(lru_state[set][1] == 1'b0) begin
            way_kick = 2'b01;
        end
    end
    else begin
        if(lru_state[set][2] == 1'b1) begin
            way_kick = 2'b10;
        end
        if(lru_state[set][2] == 1'b0) begin
            way_kick = 2'b11;
        end
    end
end

    

endmodule
