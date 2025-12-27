module SPMV(
    clk, 
	rst_n, 
	// input 
	in_valid, 
	weight_valid, 
	in_row, 
	in_col, 
	in_data, 
	// output
	out_valid, 
	out_row, 
	out_data, 
	out_finish
);

//---------------------------------------------------------------------
//   INPUT AND OUTPUT DECLARATION                         
//---------------------------------------------------------------------
input clk, rst_n; 
// input
input in_valid, weight_valid; 
input [4:0] in_row, in_col; 
input [7:0] in_data; 
// output 
output logic out_valid; 
output logic [4:0] out_row; 
output logic [20:0] out_data; 
output logic out_finish; 

//---------------------------------------------------------------------
//   LOGIC DECLARATION
//---------------------------------------------------------------------
parameter len=17;
// signal
logic mul_valid;

// Store input logic
logic [7:0] invalue [13:0];
logic [7:0] invalue_next [13:0];
logic [4:0] addr[13:0];
logic [4:0] addr_next[13:0];
logic invalue_have[13:0];

// Calculate
logic [7:0] valueA, valueA_next, valueB;
logic [4:0] row_old;
logic [len:0]sum;
//logic [15:0] product, product_next;
logic [13:0]chooseinvalue;

// Store sum
logic [len:0] sumst[31:0];
logic [len:0] sumst_next[31:0];
logic [31:0] hasvalue_n;
logic [4:0]  needincomp;


// output
logic out_valid_next;
logic [4:0]out_row_next;
logic out_ready;


//fuck
logic [4:0] out_row_n;
logic [len:0] out_data_n;
logic out_valid_f;
logic out_finish_n;

//---------------------------------------------------------------------
//   Your design                        
//---------------------------------------------------------------------

// signal
assign mul_valid = weight_valid || out_ready ;


// Store input
assign invalue_next[0] 	= in_valid? in_data	: (mul_valid? invalue[0] : 0);
assign addr_next[0] 	= in_valid? in_row	: (mul_valid? addr[0]    : 0);
/*
always @(posedge clk, negedge rst_n) begin			//this rst_n can be deleted
	if (!rst_n) begin
		invalue[0]	<= 0;
		addr[0]		<= 0;
	end
	else begin
		invalue[0] <= invalue_next[0];
		addr[0]		<= addr_next[0];
	end
end
*/
always @(posedge clk) begin			//this rst_n can be deleted

	invalue[0] <= invalue_next[0];
	addr[0]		<= addr_next[0];
end
assign invalue_have[0] = |invalue[0] ;

genvar i;
generate 
	for ( i=1; i < 14; i=i+1) begin : storing_in
		// for Storing input (In in_vector state)
		assign invalue_next[i] 	= in_valid? invalue[i-1]	: (mul_valid? invalue[i] : 0);
		assign addr_next[i] 	= in_valid? addr[i-1]		: (mul_valid? addr[i]    : 0);
		always @(posedge clk) begin
			invalue[i] <= invalue_next[i];
			addr[i]		<= addr_next[i];
		end
		assign invalue_have[i] = |invalue[i] ;
	end
endgenerate 


// Calculate(Preventing from input external delay)
genvar ii;
generate 
	for (ii = 0; ii < 14; ii = ii + 1) begin :choose_invalue
		assign chooseinvalue[ii] = (in_col==addr[ii]) & invalue_have[ii];
	end
endgenerate
//assign valueA_next = invalue[in_col];			/// This is wrong
always @(*) begin
    unique case (chooseinvalue)
        14'b0000_0000_0000_01: valueA_next = invalue[0];
        14'b0000_0000_0000_10: valueA_next = invalue[1];
        14'b0000_0000_0001_00: valueA_next = invalue[2];
        14'b0000_0000_0010_00: valueA_next = invalue[3];
        14'b0000_0000_0100_00: valueA_next = invalue[4];
        14'b0000_0000_1000_00: valueA_next = invalue[5];
        14'b0000_0001_0000_00: valueA_next = invalue[6];
        14'b0000_0010_0000_00: valueA_next = invalue[7];
        14'b0000_0100_0000_00: valueA_next = invalue[8];
        14'b0000_1000_0000_00: valueA_next = invalue[9];
        14'b0001_0000_0000_00: valueA_next = invalue[10];
        14'b0010_0000_0000_00: valueA_next = invalue[11];
        14'b0100_0000_0000_00: valueA_next = invalue[12];
        14'b1000_0000_0000_00: valueA_next = invalue[13];
        14'b0000_0000_0000_00: valueA_next = 0;
    endcase
end

always @(posedge clk) begin
	valueA 	<= valueA_next;
	valueB 	<= in_data;
	row_old <= in_row;
end

assign sum = valueA * valueB + sumst[row_old] ;


// Store Sum


	always @(*) begin
		if (out_valid_f)
			sumst_next[0] = 0;
		else begin
			if (out_ready)
				sumst_next[0] = (needincomp == 0)? sum : sumst[0] ;
			else
				sumst_next[0] = 0;
		end
	end
	/*
	always @(posedge clk, negedge rst_n) begin
		if (!rst_n)
			sumst[0] <= 0;
		else 
			sumst[0] <= sumst_next[0];
	end	
	*/
	always @(posedge clk) begin
		sumst[0] <= sumst_next[0];
	end	
	assign hasvalue_n[0] = |(sumst_next[0]) ;

assign needincomp = out_valid_f? out_row_n : row_old;

genvar j;
generate 
	for ( j=1; j < 32; j=j+1) begin :store_sum
		// for Storing Sum  (In matrix state)	
		always @(*) begin
			if (out_valid_f) begin
				if (needincomp == j)
					sumst_next[j] = 0;
				else
					sumst_next[j] = sumst[j];
			end
			else begin
				if (out_ready)
					sumst_next[j] = (needincomp == j)? sum : sumst[j] ;
				else
					sumst_next[j] = 0;
			end
		end
		always @(posedge clk) begin
			sumst[j] <= sumst_next[j];
		end	
		
		assign hasvalue_n[j] = |(sumst_next[j]) ;
	end
endgenerate 
// Choose restore value to calculate

always @(posedge clk, negedge rst_n) begin
	if (!rst_n)
		out_ready <= 0;
	else
		out_ready <= weight_valid;
end



// Output

assign out_valid_next  = (out_ready  || (|hasvalue_n) ) && (!weight_valid) ;

always @(posedge clk, negedge rst_n) begin
	if (!rst_n) begin
		out_valid_f <= 0;
		out_valid <= 0;
	end
	else begin
		out_valid_f <= out_valid_next;
		out_valid <= out_valid_f;
	end
end

assign out_finish_n = out_valid_f &  !(|hasvalue_n) ;
always @(posedge clk, negedge rst_n) begin
	if (!rst_n) 
		out_finish <= 0;
	else
		out_finish <= out_finish_n;
end

always @(*) begin
    unique casez(hasvalue_n)
    32'b????_????_????_????_????_????_????_???1: out_row_next = 5'd0;
    32'b????_????_????_????_????_????_????_??10: out_row_next = 5'd1;
    32'b????_????_????_????_????_????_????_?100: out_row_next = 5'd2;
    32'b????_????_????_????_????_????_????_1000: out_row_next = 5'd3;
    32'b????_????_????_????_????_????_???1_0000: out_row_next = 5'd4;
    32'b????_????_????_????_????_????_??10_0000: out_row_next = 5'd5;
    32'b????_????_????_????_????_????_?100_0000: out_row_next = 5'd6;
    32'b????_????_????_????_????_????_1000_0000: out_row_next = 5'd7;
    32'b????_????_????_????_????_???1_0000_0000: out_row_next = 5'd8;
    32'b????_????_????_????_????_??10_0000_0000: out_row_next = 5'd9;
    32'b????_????_????_????_????_?100_0000_0000: out_row_next = 5'd10;
    32'b????_????_????_????_????_1000_0000_0000: out_row_next = 5'd11;
    32'b????_????_????_????_???1_0000_0000_0000: out_row_next = 5'd12;
    32'b????_????_????_????_??10_0000_0000_0000: out_row_next = 5'd13;
    32'b????_????_????_????_?100_0000_0000_0000: out_row_next = 5'd14;
    32'b????_????_????_????_1000_0000_0000_0000: out_row_next = 5'd15;
    32'b????_????_????_???1_0000_0000_0000_0000: out_row_next = 5'd16;
    32'b????_????_????_??10_0000_0000_0000_0000: out_row_next = 5'd17;
    32'b????_????_????_?100_0000_0000_0000_0000: out_row_next = 5'd18;
    32'b????_????_????_1000_0000_0000_0000_0000: out_row_next = 5'd19;
    32'b????_????_???1_0000_0000_0000_0000_0000: out_row_next = 5'd20;
    32'b????_????_??10_0000_0000_0000_0000_0000: out_row_next = 5'd21;
    32'b????_????_?100_0000_0000_0000_0000_0000: out_row_next = 5'd22;
    32'b????_????_1000_0000_0000_0000_0000_0000: out_row_next = 5'd23;
    32'b????_???1_0000_0000_0000_0000_0000_0000: out_row_next = 5'd24;
    32'b????_??10_0000_0000_0000_0000_0000_0000: out_row_next = 5'd25;
    32'b????_?100_0000_0000_0000_0000_0000_0000: out_row_next = 5'd26;
    32'b????_1000_0000_0000_0000_0000_0000_0000: out_row_next = 5'd27;
    32'b???1_0000_0000_0000_0000_0000_0000_0000: out_row_next = 5'd28;
    32'b??10_0000_0000_0000_0000_0000_0000_0000: out_row_next = 5'd29;
    32'b?100_0000_0000_0000_0000_0000_0000_0000: out_row_next = 5'd30;
    32'b1000_0000_0000_0000_0000_0000_0000_0000: out_row_next = 5'd31;
    32'b0000_0000_0000_0000_0000_0000_0000_0000: out_row_next = 5'd0;
endcase
end
always @(posedge clk, negedge rst_n) begin
	if (!rst_n) begin
		out_row_n <= 0;
		out_row <= 0;
	end
	else begin
		out_row_n <= out_row_next;
		out_row <= out_row_n;
	end
end

assign out_data_n = out_valid_f? sumst[out_row_n] : 0;
always @(posedge clk, negedge rst_n) begin
	if (!rst_n) 
		out_data[len:0] <= 0;
	else
		out_data[len:0] <= out_data_n;
end
assign out_data[20:len+1] = 0;
endmodule
