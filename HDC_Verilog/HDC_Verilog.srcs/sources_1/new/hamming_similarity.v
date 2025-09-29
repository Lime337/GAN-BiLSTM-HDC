`timescale 1ns / 1ps

module hamming_similarity #(
    parameter D = 1024
)(
    input  wire [D-1:0] vec_a,
    input  wire [D-1:0] vec_b,
    output reg  [15:0] similarity
);

    integer i;
    always @(*) begin
        similarity = 0;
        for (i = 0; i < D; i = i + 1) begin
            if (vec_a[i] == vec_b[i])
                similarity = similarity + 1;
        end
    end

endmodule