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

    // array<bool, 2> all_data{false, true};
    array<int, 6>  num_threads{1, 2, 3, 4, 5, 6};
    array<int, 5>  image_sizes{256, 512, 1080, 2160, 4320};

    // for (auto data : all_data) {
        for (auto threads : num_threads) {
            for (auto size : image_sizes) {
                // args.all_data = data;
                args.num_threads = threads;
                args.image_size = size;
                experiment.set_args(args);
                experiment.run();
            }
        }
    // }

    return 0;
}
