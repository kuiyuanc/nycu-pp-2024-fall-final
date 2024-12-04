// run following command in root dir of repo to start experiment:
// $ make clean; make; srun -c 6 ./bin/main > data/log.txt

#include "experiment.hpp"

auto main(int argc, char* argv[]) -> int {
    ExperimentArgs args;
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
