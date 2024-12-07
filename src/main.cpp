// run following command in root dir of repo to start experiment:
// $ make clean; make; srun -c 6 -w hpc097 ./bin/main > data/log.txt

#include <iostream>

#include "experiment.hpp"
#include "lib/util.hpp"

auto main(int argc, char* argv[]) -> int {
    auto command_line_args = util::system::parse_args(argc, argv);
    if (command_line_args.find("num-threads") == command_line_args.end()) {
        std::cerr << "Usage: ./bin/main --num-threads <threads>" << endl;
        exit(1);
    }
    
    ExperimentArgs args(command_line_args);
    Experiment     experiment;
    dct_omp::precompute_cos_cache(8);
    dct_serial::precompute_cos_cache(8);
    if (command_line_args.find("customize") == command_line_args.end()) {
        array<bool, 2>               all_data{false, true};
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
        experiment.run(args);
    }

    return 0;
}
