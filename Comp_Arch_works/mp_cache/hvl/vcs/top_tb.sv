module top_tb;
    //---------------------------------------------------------------------------------
    // Waveform generation.
    //---------------------------------------------------------------------------------

    //---------------------------------------------------------------------------------
    // TODO: Declare cache port signals:
    //---------------------------------------------------------------------------------

    // Parameters
    parameter ADDR_WIDTH = 32;
    parameter DATA_WIDTH = 256;
    parameter CACHE_LINES = 16;

    int timeout = 10000000;

     // Cache inputs
    logic [ADDR_WIDTH-1:0] ufp_addr;
    logic [3:0]            ufp_rmask;
    logic [3:0]            ufp_wmask;
    logic [ADDR_WIDTH-1:0] ufp_wdata;
    // Cache outputs
    logic [ADDR_WIDTH-1:0] ufp_rdata;
    logic                  ufp_resp;

    logic [31:0] read_data;
    
    //---------------------------------------------------------------------------------
    // TODO: Generate a clock:
    //---------------------------------------------------------------------------------
    bit clk;
    bit rst;

    // Clock generation
    always #1ns clk = ~clk;

    //---------------------------------------------------------------------------------
    // TODO: Write a task to generate rst:
    //---------------------------------------------------------------------------------
    // initial begin
    //     rst = 1'b1;
    //     repeat (2) @(posedge clk);
    //     rst <= 1'b0;
    // end

    //---------------------------------------------------------------------------------
    // TODO: Instantiate the DUT and physical memory:
    //---------------------------------------------ufp_resp------------------------------------

    mem_itf_wo_mask mem_itf(.*);
    simple_memory_256_wo_mask simple_memory(.itf(mem_itf)); // For directed testing with PROG

    // Instantiate the cache module
    cache dut (
        .clk(clk),
        .rst(rst),

        .ufp_addr(ufp_addr),
        .ufp_rmask(ufp_rmask),
        .ufp_wmask(ufp_wmask),
        .ufp_rdata(ufp_rdata),
        .ufp_wdata(ufp_wdata),
        .ufp_resp(ufp_resp),

    // memory side signals, dfp -> downward facing port
        .dfp_addr(mem_itf.addr[0]),
        .dfp_read(mem_itf.read[0]),
        .dfp_write(mem_itf.write[0]),
        .dfp_rdata(mem_itf.rdata[0]),
        .dfp_wdata(mem_itf.wdata[0]),
        .dfp_resp(mem_itf.resp[0])
    );

    //---------------------------------------------------------------------------------
    // TODO: Write tasks to test various functionalities:
    //---------------------------------------------------------------------------------
    // task read_from_cache(
    //     input logic [31:0] addr,
    //     output logic [31:0] read_data
    // );
    //     begin
    //         ufp_addr = addr;
    //         ufp_rmask = 1'b1111;
    //         ufp_wmask = 1'b0000;
    //         read_data = ufp_rdata;
    //     end
    // endtask

    //---------------------------------------------------------------------------------
    // TODO: Main initial block that calls your tasks, then calls $finish
    //---------------------------------------------------------------------------------
    

    // Test procedure
    initial begin
        $fsdbDumpfile("dump.fsdb");
        $fsdbDumpvars(0, "+all");
        rst = 1'b1;
        repeat (2) @(posedge clk);
        rst <= 1'b0;

        // // Step 2: Write data to cache
        // write_enable = 1;
        // read_enable = 0;
        // addr = 32'h0000_0000;
        // write_data = 32'hDEADBEEF;
        // #10; // Wait for the write operation

        // // Step 3: Read data from cache
        //read_from_cache(32'h0000_0000, read_data);
        ufp_addr = 'x;
        ufp_rmask = 4'b0000;
        ufp_wmask = 4'b0000;
        ufp_wdata = 'x;
        read_data = ufp_rdata;
        // Check the result
/*
        if (read_data === 32'h9a34eceb)
            $display("Test 1 Passed: Cache hit and correct data read");
        else
            $display("Test 1 Failed: Expected 32'h9a34eceb, got %h", read_data);
*/
        #40000;
        ufp_addr <= 32'h00000000;
        ufp_rmask <= 4'b1111;
        ufp_wmask <= 4'b0000;
        ufp_wdata <= '0;
        read_data <= ufp_rdata;
        // Check the result
        if (read_data === 32'h9a34eceb)
            $display("Test 1 Passed: Cache hit and correct data read");
        else
            $display("Test 1 Failed: Expected 32'h9a34eceb, got %h", read_data);

        #20000;
/*
        ufp_addr <= 32'h00000020;
        ufp_rmask <= 4'b1111;
        ufp_wmask <= 4'b0000;
        ufp_wdata <= '0;
        read_data <= ufp_rdata;
        // Check the result
        if (read_data === 32'h44444444)
            $display("Test 1 Passed: Cache hit and correct data read");
        else
            $display("Test 1 Failed: Expected 32'h9a34eceb, got %h", read_data);

        #32000;
        ufp_addr <= 32'h00000028;
        ufp_rmask <= 4'b1111;
        ufp_wmask <= 4'b0000;
        ufp_wdata <= '0;
        read_data <= ufp_rdata;
        // Check the result
        if (read_data === 32'h44444444)
            $display("Test 1 Passed: Cache hit and correct data read");
        else
            $display("Test 1 Failed: Expected 32'h9a34eceb, got %h", read_data);

        // Step 4: Write data to another address and check eviction (if applicable)
        // addr = 32'h0000_0004;
        // write_data = 32'hCAFEBABE;
        // write_enable = 1;
        // read_enable = 0;
        // #10;

        // // Step 5: Read back the new address
        // write_enable = 0;
        // read_enable = 1;
        // addr = 32'h0000_0004;
        // #10;
        
        // if (read_data === 32'hCAFEBABE && hit)
        //     $display("Test 2 Passed: Cache hit and correct data read at new address");
        // else
        //     $display("Test 2 Failed: Expected 32'hCAFEBABE, got %h, hit = %b", read_data, hit);

        // Step 6: Check for a cache miss (optional, depends on your cache behavior)
        // addr = 32'h0000_0010;  // Address not previously written to
        // read_enable = 1;
        // write_enable = 0;
        // #10;
        
        // if (!hit)
        //     $display("Test 3 Passed: Cache miss as expected for new address");
        // else
        //     $display("Test 3 Failed: Expected miss for address %h", addr);

        // Step 7: End of simulation
        //$stop;
*/
    end

    always @(posedge clk) begin
        // if (mon_itf.halt[0]) begin
        //     $finish;
        // end
        if (timeout == 0) begin
            $error("TB Error: Timed out");
            $fatal;
        end
        if (mem_itf.error != 0) begin
            repeat (2) @(posedge clk);
            $fatal;
        end
        timeout <= timeout - 1;
    end

endmodule : top_tb
