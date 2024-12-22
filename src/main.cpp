// run following command in root dir of repo to start experiment:
// $ make clean; make; srun -c 6 -w hpc097 ./bin/main > data/log.txt

#include <iostream>

#include "experiment.hpp"
#include "../lib/util.hpp"

auto main(int argc, char* argv[]) -> int {
    auto command_line_args = util::system::parse_args(argc, argv);
    if (command_line_args.find("num-threads") == command_line_args.end() || command_line_args.find("method") == command_line_args.end()) {
        std::cerr << "Usage: ./bin/main --method <method> --num-threads <threads>" << endl;
        std::cerr << "Available methods: serial, pthread, omp, cuda" << endl;
        exit(1);
    }

    ExperimentArgs args(command_line_args);
    Experiment     experiment;
    bool full = command_line_args.find("full") != command_line_args.end();

    if (args.method == "cuda") {
        dct_cuda::copy_cache_to_device();
    }

    if (command_line_args.find("customize") == command_line_args.end()) {
        array<bool, 2>               all_data{ false, full };
        array<util::image::Shape, 3> image_sizes{
            util::image::Shape{ 512,  512},
            util::image::Shape{1920, 1080},
            util::image::Shape{2560, 1440}
        };

        for (auto data : all_data) {
            for (auto size : image_sizes) {
                args.all_data   = data;
                args.image_size = size;
                experiment.run(args);
            }
        }
    } else {
        double start{CycleTimer::currentSeconds()};
        experiment.run(args);
        if (args.verbose) {
            cout << "Total time: " << std::fixed << setprecision(3) << CycleTimer::currentSeconds() - start << " s" << endl;
        }
    }

    return 0;
}
