module MAC(
	// Input signals
	clk,
	rst_n,
	in_valid,
	in_mode,
	in_act,
	in_wgt,
	// Output signals
	out_act_idx,
	out_wgt_idx,
	out_idx,
    out_valid,
	out_data,
	out_finish
);

//---------------------------------------------------------------------
//   INPUT AND OUTPUT DECLARATION                         
//---------------------------------------------------------------------
input clk, rst_n, in_valid, in_mode;
input [0:7][3:0] in_act;
input [0:8][3:0] in_wgt;
output logic [3:0] out_act_idx, out_wgt_idx, out_idx;
output logic out_valid, out_finish;
output logic [0:7][11:0] out_data;

//---------------------------------------------------------------------
//   REG AND WIRE DECLARATION                         
//---------------------------------------------------------------------

//---------------------------------------------------------------------
//   YOUR DESIGN                        
//---------------------------------------------------------------------

///////////////////////////////////////Input stage////////////////////////////////////
// Fundemental logic 
logic mode_fet, mode_fet_next ;
logic run_fet, run_fet_next;
logic [2:0] out_wgt_idx_next, out_act_idx_next, eat_row, eat_col;
logic [5:0] eat_nextinc;

assign run_fet_next = in_valid? 1 : (run_fet? !(&{eat_row, eat_col}) : 0);
always @(posedge clk, negedge rst_n) begin
	if (!rst_n)
		run_fet <= 0;
	else
		run_fet <= run_fet_next;
end 

assign mode_fet_next = in_valid ? in_mode : mode_fet;
always @(posedge clk, negedge rst_n) begin
	if (!rst_n)
		mode_fet <= 0;
	else
		mode_fet <= mode_fet_next;
end

assign eat_row = run_fet ? (mode_fet? out_wgt_idx[2:0] : out_act_idx[2:0]) : 0;
assign eat_col = run_fet ? (mode_fet? out_act_idx[2:0] : out_wgt_idx[2:0]) : 0;
assign eat_nextinc = {eat_row, eat_col} + 6'b1;
assign out_wgt_idx_next = run_fet? (mode_fet? eat_nextinc[5:3] : eat_nextinc[2:0])	:0;
assign out_act_idx_next = run_fet? (mode_fet? eat_nextinc[2:0] : eat_nextinc[5:3])	:0;

assign out_wgt_idx[3] = 1'b1;
assign out_act_idx[3] = mode_fet;
always @(posedge clk) begin
	out_wgt_idx [2:0] <= out_wgt_idx_next;
	out_act_idx [2:0] <= out_act_idx_next;
end

// Delay for convolution
logic [2:0] col_delay, row_delay;
logic run_delay;
always @(posedge clk, negedge rst_n) begin
	if (!rst_n) begin
		col_delay <= 0;
		row_delay <= 0;
		run_delay  <=0;
	end
	else begin
		col_delay <= eat_col;
		row_delay <= eat_row;
		run_delay <= run_fet;
	end
end



// Eat data
logic [2:0] col_tidy, row_tidy, col_tidy_next, row_tidy_next;
logic run_tidy, run_tidy_next;

always_comb begin
	if (mode_fet) begin
		row_tidy_next = row_delay;
		col_tidy_next = col_delay;
		run_tidy_next = run_delay;
	end
	else begin
		row_tidy_next = eat_row;
		col_tidy_next = eat_col;
		run_tidy_next = run_fet;
	end
end

always @(posedge clk, negedge rst_n) begin
	if (!rst_n) begin
	//	col_tidy <= 0;
	//	row_tidy <= 0;
		run_tidy  <=0;
	end
	else begin
	//	col_tidy <= col_tidy_next;
	//	row_tidy <= row_tidy_next;
		run_tidy <= run_tidy_next;
	end
end
always @(posedge clk) begin
		col_tidy <= col_tidy_next;
		row_tidy <= row_tidy_next;
end


logic [0:8][3:0] astore;
logic [0:8][3:0] astore_next;
logic [0:8][3:0] bstore, bstore_next;

		assign astore_next[2] 		= mode_fet? (|eat_row 			? in_act[eat_row-1]  : 0): in_act[2];
		assign astore_next[5]		= mode_fet? in_act[eat_row]								 : in_act[5];
		assign astore_next[8] 		= mode_fet? ((&eat_row)^run_fet	? in_act[eat_row+1]: 0)	 : in_act[8];
genvar i;
generate 
	for (i=0 ; i <3 ; i=i+1) begin : fetch
		assign astore_next[3*i] 	= mode_fet? (eat_col  == 3'b001 	? 0 : astore[3*i+1]) : in_act[3*i];  
		assign astore_next[3*i+1] 	= mode_fet? (!(|{eat_col, eat_row}) && run_fet ? 0	: astore[3*i+2]) : in_act[3*i+1];
		always @(posedge clk) begin
			astore[3*i] 	<= astore_next[3*i];
			astore[3*i+1] 	<= astore_next[3*i+1];
			astore[3*i+2] 	<= astore_next[3*i+2];
		end
	end
endgenerate


genvar ii;
generate 
	for (ii=0 ; ii <3 ; ii=ii+1) begin : bb
		assign bstore_next[3*ii] 	= in_wgt[3*ii] ;
		assign bstore_next[3*ii+1] 	= in_wgt[3*ii+1] ;
		assign bstore_next[3*ii+2] 	= (mode_fet&& !(|(eat_col)))? 0: in_wgt[3*ii+2];  
		always @(posedge clk) begin
			bstore[3*ii]   <= bstore_next[3*ii];
			bstore[3*ii+1] <= bstore_next[3*ii+1];
			bstore[3*ii+2] <= bstore_next[3*ii+2];
		end
	end
endgenerate


///////////////////////////////////////Calculate stage////////////////////////////////////
/// Stage 1 : product_double 
logic [2:0] row_p1, col_p1, row_p1_next, col_p1_next;
logic run_p1, run_p1_next;



always @(posedge clk, negedge rst_n) begin
	if (!rst_n)
		run_p1 <= 0;
	else 
		run_p1 <= run_tidy;
end

always @(posedge clk) begin
	row_p1 <= row_tidy;
	col_p1 <= col_tidy;	
end

logic [0:8][5:0] sum_d ;
logic [0:8][3:0] b;
logic [0:8][5:0] prod_hi, prod_hi_next, prod_lo, prod_lo_next;
logic [0:8] cin_p1;

genvar j;
generate 
	for (j = 0; j < 9; j = j+1) begin :p1
		assign sum_d[j][0] = astore[j][0];
		assign {cin_p1[j], sum_d[j][3:1]} = astore[j][3:1] + astore[j][2:0];
		assign sum_d[j][5:4] = cin_p1[j]? astore[j][3]+ 1'b1 : { 1'b0, astore[j][3]};
		
		always_comb begin
			case (bstore[j][3:2])
				2'b00: prod_hi_next[j] = 6'b0;
				2'b01: prod_hi_next[j] = {1'b0, astore[j]};
				2'b10: prod_hi_next[j] = {astore[j], 1'b0};
				2'b11: prod_hi_next[j] = sum_d[j];
				default : prod_hi_next[j] = 0;
			endcase
			case (bstore[j][1:0])
				2'b00: prod_lo_next[j] = 6'b0;
				2'b01: prod_lo_next[j] = {1'b0, astore[j]};
				2'b10: prod_lo_next[j] = {astore[j], 1'b0};
				2'b11: prod_lo_next[j] = sum_d[j];
				default : prod_lo_next[j] = 0;
			endcase
		end
		
		always @(posedge clk) begin
			prod_hi[j] <= prod_hi_next[j];
			prod_lo[j] <= prod_lo_next[j];
		end
	
	end
endgenerate


/// Stage 2 : make product 
logic [2:0] row_p, col_p;
logic run_p;

always @(posedge clk, negedge rst_n) begin
	if (!rst_n)
		run_p <= 0;
	else 
		run_p <= run_p1;
end
always @(posedge clk) begin
	row_p <= row_p1;
	col_p <= col_p1;	
end

logic [0:8][7:0] prod, prod_next;
logic [0:8] cin_p;
genvar k;
generate 
	for (k=0; k<9; k=k+1) begin : make_product
		assign prod_next[k][1:0] = prod_lo[k][1:0];
		assign {cin_p[k], prod_next[k][5:2]} = prod_lo[k][5:2] + prod_hi[k][3:0];
		assign prod_next[k][7:6] = cin_p[k]? prod_hi[k][5:4] + 2'b01 : prod_hi[k][5:4];
		always @(posedge clk)
			prod[k] <= prod_next[k];
	end
endgenerate

/// Stage 3 : first_sum
logic [2:0] row_s1, col_s1;
logic run_s1;
logic [7:0] prod_s1;

always @(posedge clk, negedge rst_n) begin
	if (!rst_n)
		run_s1 <= 0;
	else 
		run_s1 <= run_p;
end
always @(posedge clk) begin
	row_s1 <= row_p;
	col_s1 <= col_p;
	prod_s1<= prod[8];
end

logic [0:3][8:0] sum_s1, sum_s1_next;
genvar m;
generate
	for (m=0; m<4; m=m+1) begin : make_sum
		assign sum_s1_next[m] = prod[2*m] + prod[2*m+1];
		always @(posedge clk) begin
			sum_s1[m] <= sum_s1_next[m];
		end
	end
endgenerate

/// Stage 4 : Second_sum
logic [2:0] row_s2, col_s2;
logic run_s2;
logic [7:0] prod_s2;

always @(posedge clk, negedge rst_n) begin
	if (!rst_n)
		run_s2 <= 0;
	else 
		run_s2 <= run_s1;
end
always @(posedge clk) begin
	row_s2 <= row_s1;
	col_s2 <= col_s1;
	prod_s2<= prod_s1;
end

logic [0:1][9:0] sum_s2, sum_s2_next;
assign sum_s2_next[0] = sum_s1[0] + sum_s1[1];
assign sum_s2_next[1] = sum_s1[2] + sum_s1[3];
always @(posedge clk) begin
	sum_s2[0] <= sum_s2_next[0];
	sum_s2[1] <= sum_s2_next[1];
end


/// Stage 5 :  Third Sum  (Control signal is for Convolution)
logic [2:0] row_s3, col_s3;
logic run_s3;
logic [7:0] prod_s3;

always @(posedge clk, negedge rst_n) begin
	if (!rst_n)
		run_s3 <= 0;
	else 
		run_s3 <= run_s2;
end
always @(posedge clk) begin
	row_s3 <= row_s2;
	col_s3 <= col_s2;
	prod_s3<= prod_s2;
end

logic [10:0] sum_s3, sum_s3_next;
assign sum_s3_next = sum_s2[0] + sum_s2[1];
always @(posedge clk) begin
	sum_s3 <= sum_s3_next;
end

/// Stage 6(Addition) : For the ninth element of product (Only using in convolution mode)
logic [10:0] sum_conv;
assign sum_conv = sum_s3 + prod_s3;


///////////////////////////////////////Output stage////////////////////////////////////
logic out_valid_next, out_finish_next;
logic [2:0]out_idx_next, col_choose;
logic [0:7][10:0] out_data_next;
logic [10:0] sum_choose;

assign out_valid_next 	= mode_fet? run_s3 : run_s2;
//assign out_finish_next	= out_valid & !(mode_fet? run_s2 : run_s1); 
assign out_idx_next		= mode_fet? row_s3 : row_s2;
assign col_choose 		= mode_fet? col_s3 : col_s2;
assign sum_choose		= mode_fet? sum_conv : sum_s3_next;

assign out_idx[3] = 0;
always @(posedge clk, negedge rst_n) begin
	if (!rst_n) begin
		out_valid 	<= 0;
//		out_finish	<= 0;
		out_idx[2:0]<=0;
	end
	else begin
		out_valid 	<= out_valid_next;
//		out_finish	<= out_finish_next;
		out_idx[2:0]<= out_idx_next;
	end
end 
assign out_finish = out_valid && !(mode_fet? run_s3: run_s2);


genvar p;
generate 
	for ( p =0; p < 7; p = p+1) begin : out_data_yeah
		assign out_data_next[p] = out_data[p+1]; 
		
		assign out_data[p][11] = 0;
		always @(posedge clk, negedge rst_n) begin
			if (!rst_n) 
				out_data[p][10:0] <= 0;
			else
				out_data[p][10:0] <= out_data_next[p];
		end
	end
endgenerate
		assign out_data_next[7] = sum_choose; 
		assign out_data[7][11] = 0;
		always @(posedge clk, negedge rst_n) begin
			if (!rst_n) 
				out_data[7][10:0] <= 0;
			else
				out_data[7][10:0] <= out_data_next[7];
		end

endmodule