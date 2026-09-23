#include <torch/extension.h>
#include <unordered_map>
#include <string>
#include <vector>
#include "material_histograms.h"

// Forward declaration of the CUDA function
torch::Tensor propagate_muons_cuda(torch::Tensor muon_data, torch::Tensor loss_hists, torch::Tensor bin_widths, torch::Tensor bin_centers, int num_steps);

//torch::Tensor propagate_muons_cuda(torch::Tensor input1, torch::Tensor input2, int num_steps);
// Define a wrapper function for the propagate_muons operation
torch::Tensor propagate_muons(torch::Tensor muon_data, torch::Tensor loss_hists, torch::Tensor bin_widths, torch::Tensor bin_centers, int num_steps) {
    // Check that inputs are CUDA tensors
    if (!muon_data.is_cuda() || !loss_hists.is_cuda() || !bin_widths.is_cuda() || !bin_centers.is_cuda()) {
        throw std::runtime_error("All tensors must be CUDA tensors.");
    }

    // Call the CUDA implementation
    return propagate_muons_cuda(muon_data, loss_hists, bin_widths, bin_centers, num_steps);
}



// Forward declaration of the CUDA function for uniform field per ARB8
void propagate_muons_with_alias_sampling_cuda_uniform_field(
    torch::Tensor muon_data_positions,
    torch::Tensor muon_data_momenta,
    torch::Tensor charges,
    const std::vector<MaterialHistograms>& material_histograms,
    torch::Tensor arb8s,
    torch::Tensor hashed3d_arb8s_cells,
    torch::Tensor hashed3d_arb8s_indices,
    torch::Tensor arb8s_fields,
    torch::Tensor cavern_params,
    bool use_symmetry,
    float sensitive_plane_z,
    float kill_at,
    int num_steps,
    float step_length_fixed,
    int seed,
    const float* phi_in,
    const float* dpz_in,
    const float* dsd_in
);

// Forward declaration of the AIS sample-only CUDA launcher (uniform field). Draws the per-step
// scatter magnitudes (Δp_z -> dpz_out, pt -> dsd_out) without moving the muons.
void sample_scatter_cuda_uniform_field(
    torch::Tensor muon_data_positions,
    torch::Tensor muon_data_momenta,
    const std::vector<MaterialHistograms>& material_histograms,
    torch::Tensor arb8s,
    torch::Tensor hashed3d_arb8s_cells,
    torch::Tensor hashed3d_arb8s_indices,
    torch::Tensor arb8s_fields,
    torch::Tensor cavern_params,
    bool use_symmetry,
    int seed,
    float* dpz_out,
    float* dsd_out
);

// Forward declaration of the CUDA function for magnetic field map
void propagate_muons_with_alias_sampling_cuda(
    torch::Tensor muon_data_positions,
    torch::Tensor muon_data_momenta,
    torch::Tensor charges,
    const std::vector<MaterialHistograms>& material_histograms,
    torch::Tensor arb8s,
    torch::Tensor hashed3d_arb8s_cells,
    torch::Tensor hashed3d_arb8s_indices,
    torch::Tensor magnetic_field,
    torch::Tensor magnetic_field_range,
    torch::Tensor cavern_params,
    bool use_symmetry,
    float sensitive_plane_z,
    float kill_at,
    int num_steps,
    float step_length_fixed,
    int seed
);

// Helper function to convert Python dict to vector of MaterialHistograms
std::vector<MaterialHistograms> convert_material_dict(py::dict material_dict) {
    // We expect keys in a specific order: "iron" first, then "concrete"
    // The CUDA kernel relies on this order (index 0 = iron, index 1 = concrete)
    std::vector<std::string> material_order = {"iron", "concrete"};
    std::vector<MaterialHistograms> result;
    
    for (const auto& material_name : material_order) {
        if (!material_dict.contains(material_name)) {
            throw std::runtime_error("Missing material in dict: " + material_name);
        }
        
        py::list hist_list = py::cast<py::list>(material_dict[py::str(material_name)]);
        
        if (hist_list.size() != 6) {
            throw std::runtime_error("Each material must have exactly 6 histogram tensors. Got " + 
                                     std::to_string(hist_list.size()) + " for " + material_name);
        }
        
        MaterialHistograms mh;
        mh.probability_table = py::cast<torch::Tensor>(hist_list[0]);
        mh.alias_table = py::cast<torch::Tensor>(hist_list[1]);
        mh.bin_centers_first_dim = py::cast<torch::Tensor>(hist_list[2]);
        mh.bin_centers_second_dim = py::cast<torch::Tensor>(hist_list[3]);
        mh.bin_widths_first_dim = py::cast<torch::Tensor>(hist_list[4]);
        mh.bin_widths_second_dim = py::cast<torch::Tensor>(hist_list[5]);
        
        result.push_back(mh);
    }
    
    return result;
}

//torch::Tensor propagate_muons_with_alias_sampling_cuda(torch::Tensor input1, torch::Tensor input2, int num_steps);
// Define a wrapper function for the propagate_muons_with_alias_sampling operation (uniform field version)
void propagate_muons_with_alias_sampling_uniform_field(
    torch::Tensor muon_data_positions,
    torch::Tensor muon_data_momenta,
    torch::Tensor charges,
    py::dict material_histograms_dict,  // Python dict: {'iron': [...], 'concrete': [...]}
    torch::Tensor arb8s,
    torch::Tensor hashed3d_arb8s_cells,
    torch::Tensor hashed3d_arb8s_indices,
    torch::Tensor arb8s_fields,
    torch::Tensor cavern_params,
    bool use_symmetry,
    float sensitive_plane_z,
    float kill_at,
    int num_steps,
    float step_length_fixed,
    int seed
) {
    // Convert Python dict to C++ vector
    std::vector<MaterialHistograms> material_histograms = convert_material_dict(material_histograms_dict);
    
    // Check that inputs are CUDA tensors
    if (!muon_data_positions.is_cuda() || !muon_data_momenta.is_cuda() ||
        !arb8s.is_cuda() || !arb8s_fields.is_cuda() || !cavern_params.is_cuda()) {
        throw std::runtime_error("All tensors must be CUDA tensors.");
    }
    
    // Check material histogram tensors
    for (size_t i = 0; i < material_histograms.size(); ++i) {
        const auto& mh = material_histograms[i];
        if (!mh.probability_table.is_cuda() || !mh.alias_table.is_cuda() ||
            !mh.bin_centers_first_dim.is_cuda() || !mh.bin_centers_second_dim.is_cuda() ||
            !mh.bin_widths_first_dim.is_cuda() || !mh.bin_widths_second_dim.is_cuda()) {
            throw std::runtime_error("All material histogram tensors must be CUDA tensors.");
        }
    }

    // Call the CUDA implementation for uniform field.
    // phi_in = nullptr -> the kernel samples the scatter azimuth internally (original behaviour).
    // dpz_in = dsd_in = nullptr -> the kernel samples the scatter magnitudes internally too.
    propagate_muons_with_alias_sampling_cuda_uniform_field(
        muon_data_positions,
        muon_data_momenta,
        charges,
        material_histograms,
        arb8s,
        hashed3d_arb8s_cells,
        hashed3d_arb8s_indices,
        arb8s_fields,
        cavern_params,
        use_symmetry,
        sensitive_plane_z,
        kill_at,
        num_steps,
        step_length_fixed,
        seed,
        nullptr,
        nullptr,
        nullptr
    );
}

// One-step variant for adaptive importance sampling: the per-muon azimuthal scatter
// angle is supplied externally via `phi` (the proposal-policy knob) instead of being
// sampled uniformly inside the kernel. All other physics (histogram magnitude sampling,
// material lookup, RK4 field integration, termination) is identical to the function above.
// `phi` must be a CUDA float32 tensor of length N (one angle per muon); intended for num_steps == 1.
void propagate_muons_one_step_uniform_field(
    torch::Tensor muon_data_positions,
    torch::Tensor muon_data_momenta,
    torch::Tensor charges,
    py::dict material_histograms_dict,  // Python dict: {'iron': [...], 'concrete': [...]}
    torch::Tensor arb8s,
    torch::Tensor hashed3d_arb8s_cells,
    torch::Tensor hashed3d_arb8s_indices,
    torch::Tensor arb8s_fields,
    torch::Tensor cavern_params,
    bool use_symmetry,
    float sensitive_plane_z,
    float kill_at,
    int num_steps,
    float step_length_fixed,
    int seed,
    torch::Tensor phi,
    torch::Tensor dpz,   // pre-sampled Δp_z per muon, or empty tensor -> sample in-kernel
    torch::Tensor dsd    // pre-sampled pt per muon, or empty tensor -> sample in-kernel
) {
    // Convert Python dict to C++ vector
    std::vector<MaterialHistograms> material_histograms = convert_material_dict(material_histograms_dict);

    // Check that inputs are CUDA tensors
    if (!muon_data_positions.is_cuda() || !muon_data_momenta.is_cuda() ||
        !arb8s.is_cuda() || !arb8s_fields.is_cuda() || !cavern_params.is_cuda() || !phi.is_cuda()) {
        throw std::runtime_error("All tensors must be CUDA tensors.");
    }
    TORCH_CHECK(phi.scalar_type() == torch::kFloat32, "phi must be a float32 tensor");
    TORCH_CHECK(phi.numel() == muon_data_positions.size(0),
                "phi must have exactly one entry per muon (length N).");

    // Check material histogram tensors
    for (size_t i = 0; i < material_histograms.size(); ++i) {
        const auto& mh = material_histograms[i];
        if (!mh.probability_table.is_cuda() || !mh.alias_table.is_cuda() ||
            !mh.bin_centers_first_dim.is_cuda() || !mh.bin_centers_second_dim.is_cuda() ||
            !mh.bin_widths_first_dim.is_cuda() || !mh.bin_widths_second_dim.is_cuda()) {
            throw std::runtime_error("All material histogram tensors must be CUDA tensors.");
        }
    }

    torch::Tensor phi_c = phi.contiguous();

    // Optional pre-sampled scatter magnitudes: when both dpz and dsd are non-empty the kernel
    // uses them and skips the histogram draw (the AIS "sample first, decide azimuth" path);
    // empty tensors -> nullptr -> the kernel samples the magnitudes internally as before.
    const float* dpz_ptr = nullptr;
    const float* dsd_ptr = nullptr;
    torch::Tensor dpz_c, dsd_c;
    if (dsd.defined() && dsd.numel() > 0) {
        TORCH_CHECK(dpz.defined() && dpz.numel() == phi.numel(),
                    "dpz must have one entry per muon (length N) when supplied.");
        TORCH_CHECK(dsd.numel() == phi.numel(),
                    "dsd must have one entry per muon (length N) when supplied.");
        TORCH_CHECK(dpz.is_cuda() && dsd.is_cuda(), "dpz/dsd must be CUDA tensors when supplied.");
        TORCH_CHECK(dpz.scalar_type() == torch::kFloat32 && dsd.scalar_type() == torch::kFloat32,
                    "dpz/dsd must be float32 tensors.");
        dpz_c = dpz.contiguous();
        dsd_c = dsd.contiguous();
        dpz_ptr = dpz_c.data_ptr<float>();
        dsd_ptr = dsd_c.data_ptr<float>();
    }

    propagate_muons_with_alias_sampling_cuda_uniform_field(
        muon_data_positions,
        muon_data_momenta,
        charges,
        material_histograms,
        arb8s,
        hashed3d_arb8s_cells,
        hashed3d_arb8s_indices,
        arb8s_fields,
        cavern_params,
        use_symmetry,
        sensitive_plane_z,
        kill_at,
        num_steps,
        step_length_fixed,
        seed,
        phi_c.data_ptr<float>(),
        dpz_ptr,
        dsd_ptr
    );
}

// AIS sample-only entry point (uniform field): draws the per-step scatter magnitudes
// (Δp_z -> dpz_out, pt -> dsd_out) from the alias histograms at each muon's current
// (position, momentum) WITHOUT moving the muon. Pair with propagate_muons_one_step_uniform_field
// (passing dpz_out/dsd_out as its dpz/dsd) so the agent can condition its azimuth on the sampled
// magnitude. dpz_out/dsd_out are written in place and must be contiguous float32 CUDA tensors of
// length N.
void sample_scatter_uniform_field(
    torch::Tensor muon_data_positions,
    torch::Tensor muon_data_momenta,
    py::dict material_histograms_dict,  // Python dict: {'iron': [...], 'concrete': [...]}
    torch::Tensor arb8s,
    torch::Tensor hashed3d_arb8s_cells,
    torch::Tensor hashed3d_arb8s_indices,
    torch::Tensor arb8s_fields,
    torch::Tensor cavern_params,
    bool use_symmetry,
    int seed,
    torch::Tensor dpz_out,
    torch::Tensor dsd_out
) {
    std::vector<MaterialHistograms> material_histograms = convert_material_dict(material_histograms_dict);

    if (!muon_data_positions.is_cuda() || !muon_data_momenta.is_cuda() ||
        !arb8s.is_cuda() || !arb8s_fields.is_cuda() || !cavern_params.is_cuda() ||
        !dpz_out.is_cuda() || !dsd_out.is_cuda()) {
        throw std::runtime_error("All tensors must be CUDA tensors.");
    }
    TORCH_CHECK(dpz_out.scalar_type() == torch::kFloat32 && dsd_out.scalar_type() == torch::kFloat32,
                "dpz_out/dsd_out must be float32 tensors.");
    TORCH_CHECK(dpz_out.is_contiguous() && dsd_out.is_contiguous(),
                "dpz_out/dsd_out must be contiguous (written in place).");
    TORCH_CHECK(dpz_out.numel() == muon_data_positions.size(0) &&
                dsd_out.numel() == muon_data_positions.size(0),
                "dpz_out/dsd_out must have one entry per muon (length N).");

    for (size_t i = 0; i < material_histograms.size(); ++i) {
        const auto& mh = material_histograms[i];
        if (!mh.probability_table.is_cuda() || !mh.alias_table.is_cuda() ||
            !mh.bin_centers_first_dim.is_cuda() || !mh.bin_centers_second_dim.is_cuda() ||
            !mh.bin_widths_first_dim.is_cuda() || !mh.bin_widths_second_dim.is_cuda()) {
            throw std::runtime_error("All material histogram tensors must be CUDA tensors.");
        }
    }

    sample_scatter_cuda_uniform_field(
        muon_data_positions,
        muon_data_momenta,
        material_histograms,
        arb8s,
        hashed3d_arb8s_cells,
        hashed3d_arb8s_indices,
        arb8s_fields,
        cavern_params,
        use_symmetry,
        seed,
        dpz_out.data_ptr<float>(),
        dsd_out.data_ptr<float>()
    );
}

// Define a wrapper function for the propagate_muons_with_alias_sampling operation (field map version)
void propagate_muons_with_alias_sampling(
    torch::Tensor muon_data_positions,
    torch::Tensor muon_data_momenta,
    torch::Tensor charges,
    py::dict material_histograms_dict,  // Python dict: {'iron': [...], 'concrete': [...]}
    torch::Tensor arb8s,
    torch::Tensor hashed3d_arb8s_cells,
    torch::Tensor hashed3d_arb8s_indices,
    torch::Tensor magnetic_field,
    torch::Tensor magnetic_field_range,
    torch::Tensor cavern_params,
    bool use_symmetry,
    float sensitive_plane_z,
    float kill_at,
    int num_steps,
    float step_length_fixed,
    int seed
) {
    // Convert Python dict to C++ vector
    std::vector<MaterialHistograms> material_histograms = convert_material_dict(material_histograms_dict);
    
    // Check that inputs are CUDA tensors
    if (!muon_data_positions.is_cuda() || !muon_data_momenta.is_cuda() ||
        !magnetic_field.is_cuda() || !arb8s.is_cuda() || !cavern_params.is_cuda()) {
        throw std::runtime_error("All tensors must be CUDA tensors.");
    }
    
    // Check material histogram tensors
    for (size_t i = 0; i < material_histograms.size(); ++i) {
        const auto& mh = material_histograms[i];
        if (!mh.probability_table.is_cuda() || !mh.alias_table.is_cuda() ||
            !mh.bin_centers_first_dim.is_cuda() || !mh.bin_centers_second_dim.is_cuda() ||
            !mh.bin_widths_first_dim.is_cuda() || !mh.bin_widths_second_dim.is_cuda()) {
            throw std::runtime_error("All material histogram tensors must be CUDA tensors.");
        }
    }

    // Call the CUDA implementation for field map
    propagate_muons_with_alias_sampling_cuda(
        muon_data_positions,
        muon_data_momenta,
        charges,
        material_histograms,
        arb8s,
        hashed3d_arb8s_cells,
        hashed3d_arb8s_indices,
        magnetic_field,
        magnetic_field_range,
        cavern_params,
        use_symmetry,
        sensitive_plane_z,
        kill_at,
        num_steps,
        step_length_fixed,
        seed
    );
}


void add_cuda(torch::Tensor x, torch::Tensor y, torch::Tensor out);

// Define a wrapper function that can be called from Python for addition
void add(torch::Tensor x, torch::Tensor y, torch::Tensor out) {
    // Check that inputs are on the same CUDA device
    if (!x.is_cuda() || !y.is_cuda() || !out.is_cuda()) {
        throw std::runtime_error("All tensors must be CUDA tensors.");
    }
    add_cuda(x, y, out);
}
// Register the functions as PyTorch extensions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Add two tensors (CUDA)");
    m.def("propagate_muons", &propagate_muons, "Propagate muons with two input tensors (CUDA)");
    m.def("propagate_muons_with_alias_sampling", &propagate_muons_with_alias_sampling, 
          "Propagate muons with magnetic field map (CUDA)");
    m.def("propagate_muons_with_alias_sampling_uniform_field", &propagate_muons_with_alias_sampling_uniform_field,
          "Propagate muons with uniform magnetic field per ARB8 (CUDA)");
    m.def("propagate_muons_one_step_uniform_field", &propagate_muons_one_step_uniform_field,
          "Single-step uniform-field propagation with externally supplied azimuthal scatter angle (CUDA, for AIS)");
    m.def("sample_scatter_uniform_field", &sample_scatter_uniform_field,
          "Sample per-step scatter magnitudes (dpz, pt) from the alias histograms without moving the muons (CUDA, for AIS)");
}
