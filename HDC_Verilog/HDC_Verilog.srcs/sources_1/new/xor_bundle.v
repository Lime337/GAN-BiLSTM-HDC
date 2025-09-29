`timescale 1ns / 1ps

module xor_bundle #(
    parameter D = 1024,
    parameter CHUNK = 10
)(
    input  wire [CHUNK*D-1:0] input_lines_flat,
    input  wire [CHUNK*D-1:0] position_vectors_flat,
    output reg  [D-1:0] bundled_vector
);

    integer i, j;
    integer count;
    reg bit_in, bit_pos;

    always @(*) begin
        for (i = 0; i < D; i = i + 1) begin
            count = 0;
            for (j = 0; j < CHUNK; j = j + 1) begin
                bit_in  = input_lines_flat[j*D + i];
                bit_pos = position_vectors_flat[j*D + i];
                count   = count + (bit_in ^ bit_pos);  //XOR binding
            end
            bundled_vector[i] = (count >= (CHUNK / 2)) ? 1'b1 : 1'b0;
        end
    end

endmodule
