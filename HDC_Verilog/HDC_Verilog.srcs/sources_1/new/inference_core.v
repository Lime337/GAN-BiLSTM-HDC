`timescale 1ns / 1ps

module inference_core #(
    parameter D = 1024,
    parameter CHUNK = 10
)(
    input  wire [CHUNK*D-1:0] input_lines_flat,
    input  wire [CHUNK*D-1:0] position_vectors_flat,
    input  wire [D-1:0] benign_class_vector,
    input  wire [D-1:0] malware_class_vector,
    output wire [15:0] sim_benign,
    output wire [15:0] sim_malware,
    output wire        prediction
);

    wire [D-1:0] test_vector;

    xor_bundle #(.D(D), .CHUNK(CHUNK)) bundler (
        .input_lines_flat(input_lines_flat),
        .position_vectors_flat(position_vectors_flat),
        .bundled_vector(test_vector)
    );

    hamming_similarity #(.D(D)) simB (
        .vec_a(test_vector),
        .vec_b(benign_class_vector),
        .similarity(sim_benign)
    );

    hamming_similarity #(.D(D)) simM (
        .vec_a(test_vector),
        .vec_b(malware_class_vector),
        .similarity(sim_malware)
    );

    assign prediction = (sim_malware > sim_benign) ? 1'b1 : 1'b0;

endmodule
