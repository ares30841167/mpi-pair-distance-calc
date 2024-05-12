#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>

#define N                   10008
#define TOP_RANKING         5
#define INVAILD_IDX         -1
#define INVAILD_DISTANCE    -1
#define MIN_COORD_POINT     0.0
#define MAX_COORD_POINT     1000.0
#define MAX_PROCESS_NUM     4

struct Coord {
    double x;
    double y;
};

struct Rank {
    int     c1_idx;
    int     c2_idx;
    double  distance;
};

/**
 * @brief Get the number of MPI processes in the communicator MPI_COMM_WORLD.
 * 
 * @return The number of MPI processes.
 */
int mpi_get_nprocs() {
    int nprocs;

    // Get the size of the MPI communicator MPI_COMM_WORLD
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    return nprocs;
}

/**
 * @brief Get the rank of the current MPI process in the communicator MPI_COMM_WORLD.
 * 
 * @return The rank of the current MPI process.
 */
int mpi_get_current_rank() {
    int myrank;

    // Get the rank of the current MPI process
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    return myrank;
}

/**
 * @brief Create an MPI datatype for the struct Rank.
 * 
 * @return The MPI datatype for struct Rank.
 */
MPI_Datatype mpi_create_rank_type() {
    // Define the number of items in the struct Rank
    const int nitems = 3;

    // Define the block lengths for each item in the struct Rank
    int blocklengths[3] = {1, 1, 1};

    // Define the MPI datatypes for each item in the struct Rank
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};

    // Define the MPI datatype to be created
    MPI_Datatype mpi_rank_type;

    // Define the offsets of each item in the struct Rank
    MPI_Aint offsets[3];

    // Calculate the offsets of each item in the struct Rank
    offsets[0] = offsetof(struct Rank, c1_idx);
    offsets[1] = offsetof(struct Rank, c2_idx);
    offsets[2] = offsetof(struct Rank, distance);

    // Create the MPI datatype for the struct Rank
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_rank_type);

    // Commit the MPI datatype for use
    MPI_Type_commit(&mpi_rank_type);

    // Return the created MPI datatype for struct Rank
    return mpi_rank_type;
}

/**
 * @brief Print debug information for the coordinate data.
 * 
 * This function prints the coordinates of each point for debugging purposes,
 * including the MPI process rank, index, and the x, y coordinates.
 * 
 * @param c Pointer to an array of Coord structs containing coordinate data.
 */
void debug_print_coord_info(struct Coord* c) {
    // Get the rank of the current MPI process
    int myrank = mpi_get_current_rank();
    
    // Iterate through each coordinate and print debug information
    for(int i = 0; i < N; ++i) {
        printf("[DEBUG] Proc %d, coord[%d] -> x: %lf, y: %lf\n", myrank, i, c[i].x, c[i].y);
    }
}

/**
 * @brief Generate a random double number within a specified range.
 * 
 * This function generates a random double number within the range specified by
 * the constants MAX_COORD_POINT and MIN_COORD_POINT using the rand() function.
 * 
 * @return A random double number within the specified range.
 */
double gen_random_number() {
    return (MAX_COORD_POINT - MIN_COORD_POINT) * rand() / (RAND_MAX + 1.0) + MIN_COORD_POINT;
}

/**
 * @brief Generate random coordinate data.
 * 
 * This function dynamically allocates memory for an array of Coord structs,
 * generates random x and y coordinates for each point in the array using the
 * gen_random_number function, and returns a pointer to the array.
 * 
 * @return Pointer to an array of Coord structs containing generated coordinate data.
 */
struct Coord* coord_gen_random_data() {
    // Allocate memory for the array of Coord structs
    struct Coord* pRd = malloc(sizeof(struct Coord) * N);
    
    // Generate random x and y coordinates for each point in the array
    for(int i = 0; i < N; ++i) {
        pRd[i].x = gen_random_number();
        pRd[i].y = gen_random_number();
    }

    // Return a pointer to the array
    return pRd;
}

/**
 * @brief Calculate the Euclidean distance between two coordinates.
 * 
 * This function calculates the Euclidean distance between two coordinates
 * using the formula: distance = sqrt((x2 - x1)^2 + (y2 - y1)^2).
 * 
 * @param c1 Pointer to the first Coord struct.
 * @param c2 Pointer to the second Coord struct.
 * @return The calculated Euclidean distance.
 */
double calc_distance(struct Coord* c1, struct Coord* c2) {
    // Calculate the differences in x and y coordinates
    double d_x = c2->x - c1->x;
    double d_y = c2->y - c1->y;

    // Calculate the square of the differences and sum them up
    double squared_distance = d_x * d_x + d_y * d_y;

    // Calculate the square root of the sum to get the distance
    double distance = sqrt(squared_distance);

    // Return the calculated Euclidean distance
    return distance;
}

/**
 * @brief Initialize an array of Rank structs.
 * 
 * This function dynamically allocates memory for an array of Rank structs,
 * initializes each struct with invalid values for c1_idx, c2_idx, and distance,
 * and returns a pointer to the array.
 * 
 * @return Pointer to an array of Rank structs with initialized values.
 */
struct Rank* rank_init_ranking_arr() {
    // Allocate memory for the array of Rank structs
    struct Rank* pRa = malloc(sizeof(struct Rank) * TOP_RANKING);
    
    // Initialize each struct in the array with invalid values
    for(int i = 0; i < TOP_RANKING; ++i) {
        pRa[i].c1_idx = INVAILD_IDX;
        pRa[i].c2_idx = INVAILD_IDX;
        pRa[i].distance = INVAILD_DISTANCE;
    }

    // Return a pointer to the initialized array
    return pRa;
}

/**
 * @brief Print debug information for rank updated.
 * 
 * This function prints debug information about an updated rank, including
 * the MPI process rank, the index of the updated ranking, and the distance
 * and indices of the updated ranking.
 * 
 * @param inserted_idx Index of the updated rank.
 * @param r Pointer to the updated Rank struct.
 */
void debug_print_rank_updated_info(int inserted_idx, struct Rank* r) {
    // Get the rank of the current MPI process
    int myrank = mpi_get_current_rank();
    
    // Print debug information about the updated rank
    printf("[DEBUG] Proc %d, Ranking %d updated -> %lf  (%d, %d)\n", 
                                        myrank, inserted_idx, r->distance, r->c1_idx, r->c2_idx);
}

/**
 * @brief Check if a pair of coordinates is a duplicate in the given rank.
 * 
 * This function checks if a pair of coordinates (c1_idx, c2_idx) or (c2_idx, c1_idx)
 * already exists in the given Rank struct.
 * 
 * @param r Pointer to the Rank struct to check.
 * @param c1_idx Index of the first coordinate.
 * @param c2_idx Index of the second coordinate.
 * @return true if the pair is a duplicate, false otherwise.
 */
bool is_duplicate_pair(struct Rank* r, int c1_idx, int c2_idx) {
    // Check if the pair (c1_idx, c2_idx) or (c2_idx, c1_idx) already exists in the given Rank struct
    if (r->c1_idx == c1_idx && r->c2_idx == c2_idx)
        return true;
    if (r->c1_idx == c2_idx && r->c2_idx == c1_idx)
        return true;
    
    // If the pair is not found, return false
    return false;
}

/**
 * @brief Check if a pair of coordinates exists in the given rank array.
 * 
 * This function checks if a pair of coordinates (c1_idx, c2_idx) or (c2_idx, c1_idx)
 * already exists in any of the Rank structs within the given array of ranks.
 * 
 * @param r Pointer to the array of Rank structs to check.
 * @param c1_idx Index of the first coordinate.
 * @param c2_idx Index of the second coordinate.
 * @return true if the pair exists in any of the ranks, false otherwise.
 */
bool is_pair_existed(struct Rank* r, int c1_idx, int c2_idx) {
    // Iterate through each Rank struct in the array
    for(int i = 0; i < TOP_RANKING; ++i) {
        // Check if the pair (c1_idx, c2_idx) or (c2_idx, c1_idx) exists in the current Rank struct
        if(is_duplicate_pair(&r[i], c1_idx, c2_idx))
            return true;
    }
    // If the pair is not found in any of the Rank structs, return false
    return false;
}

/**
 * @brief Update the ranking with a new pair of coordinates and their distance.
 * 
 * This function updates the ranking with a new pair of coordinates (c1_idx, c2_idx) 
 * and their corresponding distance. It checks if the pair already exists in the 
 * ranking, and if not, it updates the ranking by inserting the new pair at the 
 * appropriate position based on the distance.
 * 
 * @param r Pointer to the array of Rank structs representing the ranking.
 * @param c1_idx Index of the first coordinate.
 * @param c2_idx Index of the second coordinate.
 * @param distance Distance between the coordinates.
 * @return Pointer to the updated array of Rank structs representing the ranking.
 */
struct Rank* rank_update_ranking(struct Rank* r, int c1_idx, int c2_idx, double distance) {
    // Check if the pair of coordinates already exists in the ranking
    if(is_pair_existed(r, c1_idx, c2_idx))
        return r; // If the pair exists, no need to update the ranking, return the original ranking

    // Iterate through the ranking to find the appropriate position to insert the new pair
    for(int i = 0; i < TOP_RANKING; ++i) {
        // If the current position in the ranking is empty or the new distance is smaller than the existing one
        if(r[i].distance == INVAILD_DISTANCE || distance <= r[i].distance) {
            // If the current position is not empty, shift the existing pairs to make space
            if(distance <= r[i].distance) {
                for(int cpy_ptr = TOP_RANKING - 1; cpy_ptr >= i + 1; --cpy_ptr) {
                    r[cpy_ptr] = r[cpy_ptr - 1];
                }
            }

            // Create a new Rank struct for the new pair of coordinates and distance
            struct Rank new_rank = {
                c1_idx,
                c2_idx,
                distance
            };
            // Insert the new pair at the appropriate position in the ranking
            r[i] = new_rank;

            // Print debug information if DEBUG mode is enabled
            #ifdef DEBUG
            debug_print_rank_updated_info(i, &new_rank);
            #endif

            // Exit the loop after updating the ranking
            break;
        }
    }

    // Return the pointer to the updated array of Rank structs representing the ranking
    return r;
}

/**
 * @brief Calculate distances and generate rankings for all coordinates within the assigned chunk.
 * 
 * This function calculates the distances between all pairs of coordinates within the assigned
 * chunk for the current MPI process and generates rankings based on these distances.
 * 
 * @param nprocs Number of MPI processes.
 * @param myrank Rank of the current MPI process.
 * @param c Pointer to the array of Coord structs containing coordinate data.
 * @return Pointer to the array of Rank structs representing the generated rankings.
 */
struct Rank* calc_all_distance_and_ranking(int nprocs, int myrank, struct Coord* c) {
    // Initialize an array of Rank structs to store rankings
    struct Rank* rank = rank_init_ranking_arr();

    // Calculate the size of the chunk assigned to the current MPI process
    int chunk_size = N / nprocs;
    // Calculate the starting index of the chunk for the current MPI process
    int start_idx = chunk_size * myrank;

    // Iterate over each coordinate within the assigned chunk
    for(int chunk_ptr = start_idx; chunk_ptr < start_idx + chunk_size; ++chunk_ptr) {
        // Calculate distances between the current coordinate and all other coordinates
        for(int i = 0; i < N; ++i) {
            // Skip calculation if the current coordinate is the same as the one being compared
            if(chunk_ptr == i) continue;
            // Calculate the distance between the current coordinate and the other coordinate
            double distance = calc_distance(&c[chunk_ptr], &c[i]);
            // Update the ranking based on the calculated distance
            rank_update_ranking(rank, chunk_ptr, i, distance);
        }
    }

    // Return the pointer to the array of Rank structs representing the generated rankings
    return rank;
}

/**
 * @brief Print debug information for received rankings.
 * 
 * This function prints debug information for the received rankings, including
 * the index of each ranking, the indices of the paired coordinates, and the
 * corresponding distance.
 * 
 * @param r Pointer to the array of Rank structs representing received rankings.
 */
void debug_rank_print_recv_rank(struct Rank* r) {
    // Get the total number of MPI processes
    int nprocs = mpi_get_nprocs();
    // Calculate the total number of rankings based on the number of processes and the top ranking count
    int total_cnt = nprocs * TOP_RANKING;

    // Iterate over each received ranking and print debug information
    for (int i = 0; i < total_cnt; ++i) {
        printf("[DEBUG] Received ranking %d: c1_idx = %d, c2_idx = %d, distance = %f\n", 
                                                    i, r[i].c1_idx, r[i].c2_idx, r[i].distance);
    }
}

/**
 * @brief Check if two rankings contain the same pair of coordinates.
 * 
 * This function checks if two rankings contain the same pair of coordinates,
 * regardless of the order of the indices in each pair.
 * 
 * @param r1 Pointer to the first array of Rank structs representing a ranking.
 * @param r2 Pointer to the second array of Rank structs representing a ranking.
 * @return true if the rankings contain the same pair of coordinates, false otherwise.
 */
bool is_duplicate_rank(struct Rank* r1, struct Rank* r2) {
    // Check if the pair of coordinates in r1 matches the pair in r2
    if (r1->c1_idx == r2->c1_idx && r1->c2_idx == r2->c2_idx)
        return true;
    // Check if the pair of coordinates in r1 matches the reverse order of the pair in r2
    if (r1->c1_idx == r2->c2_idx && r1->c2_idx == r2->c1_idx)
        return true;
    // If no match is found, return false
    return false;
}

/**
 * @brief Print debug information for removed duplicate ranking.
 * 
 * This function prints debug information when a duplicate ranking is removed.
 * It includes the index of the removed ranking and the indices of the paired 
 * coordinates in both duplicate rankings.
 * 
 * @param removed_idx Index of the removed duplicate ranking.
 * @param r1 Pointer to the first array of Rank structs representing a duplicate ranking.
 * @param r2 Pointer to the second array of Rank structs representing a duplicate ranking.
 */
void debug_print_rank_duplicate_removed(int removed_idx, struct Rank* r1, struct Rank* r2) {
    // Print debug information about the removed duplicate ranking
    printf("[DEBUG] Received ranking %d removed -> r1: (%d, %d), r2: (%d, %d)\n",
                                    removed_idx, r1->c1_idx, r1->c2_idx, r2->c1_idx, r2->c2_idx);
}

/**
 * @brief Remove duplicate rankings from the array.
 * 
 * This function removes duplicate rankings from the array of Rank structs.
 * It iterates through the array and compares each pair of rankings to check
 * for duplicates. If a duplicate is found, the corresponding ranking is marked
 * as invalid by setting its distance to INVAILD_DISTANCE.
 * 
 * @param r Pointer to the array of Rank structs.
 * @return Pointer to the modified array with duplicate rankings removed.
 */
struct Rank* remove_duplicates(struct Rank* r) {
    // Get the total number of MPI processes
    int nprocs = mpi_get_nprocs();
    // Calculate the total number of rankings based on the number of processes and the top ranking count
    int total_cnt = nprocs * TOP_RANKING;

    // Iterate through the array to check for duplicate rankings
    for(int i = 0; i < total_cnt; ++i) {
        // Skip empty rankings
        if(r[i].distance == INVAILD_DISTANCE) continue;
        
        // Compare the current ranking with the rest of the rankings
        for(int j = i + 1; j < total_cnt; ++j) {
            // Skip empty rankings
            if(r[j].distance == INVAILD_DISTANCE) continue;

            // If a duplicate ranking is found, mark it as empty
            if(!is_duplicate_rank(&r[i], &r[j])) continue;

            // Print debug information if DEBUG mode is enabled
            #ifdef DEBUG
            debug_print_rank_duplicate_removed(j, &r[i], &r[j]);
            #endif

            // Replace the duplicate ranking with empty rank struct
            struct Rank empty_rank = {
                INVAILD_IDX,
                INVAILD_IDX,
                INVAILD_DISTANCE
            };
            r[j] = empty_rank;
        }
    }

    // Return the pointer to the modified array with duplicate rankings removed
    return r;
}

/**
 * @brief Comparison function for sorting rankings based on distance.
 * 
 * This function compares two rankings based on their distances.
 * It is used as a comparison function for sorting rankings in ascending order
 * of distance, with empty rankings (distance == INVAILD_DISTANCE) placed at
 * the end of the sorted array.
 * 
 * @param r1 Pointer to the first ranking to compare.
 * @param r2 Pointer to the second ranking to compare.
 * @return An integer value indicating the result of the comparison:
 *         - 0 if the distances of both rankings are equal.
 *         - 1 if the distance of the first ranking is greater.
 *         - -1 if the distance of the second ranking is greater.
 */
int compare_rank(const void* r1, const void* r2) {
    // Extract distances from the rankings
    double r1_distance = ((struct Rank*)r1)->distance;
    double r2_distance = ((struct Rank*)r2)->distance;

    // Compare distances and handle invalid rankings
    if(r1_distance == INVAILD_DISTANCE)
        return 1; // r1 is considered greater if it is invalid
    if(r2_distance == INVAILD_DISTANCE)
        return -1; // r2 is considered greater if it is invalid

    // Compare distances for valid rankings
    if(r2_distance > r1_distance)
        return -1; // r2 is considered greater
    else
        return 1; // r1 is considered greater or equal to r2
}

/**
 * @brief Print the result rankings.
 * 
 * This function prints the result rankings to the standard output.
 * It iterates through the array of Rank structs and prints each ranking's
 * distance along with the indices of the paired coordinates.
 * 
 * @param r Pointer to the array of Rank structs representing the result rankings.
 */
void print_result(struct Rank* r) {
    // Iterate through the array of Rank structs
    for(int i = 0; i < TOP_RANKING; ++i) {
        // Print the distance and indices of the paired coordinates for each ranking
        printf("%lf  (%d, %d)\n", r[i].distance, r[i].c1_idx, r[i].c2_idx);
    }
}

/**
 * @brief Perform post-processing and print the result.
 * 
 * This function performs post-processing on the received rankings, including
 * removing duplicate rankings, sorting the rankings based on distance, and
 * printing the final result to the standard output. It also measures the
 * execution time of this distance-calculating task.
 * 
 * @param received_rank Pointer to the array of Rank structs representing received rankings.
 * @return The execution time of the distance-calculating task.
 */
double post_process_and_print_result(struct Rank* received_rank) {
    // Print debug information for received rankings if DEBUG mode is enabled
    #ifdef DEBUG
    debug_rank_print_recv_rank(received_rank);
    #endif

    // Remove duplicate rankings from the received rankings
    received_rank = remove_duplicates(received_rank);

    // Get the total number of MPI processes and calculate the total count of rankings
    int nprocs = mpi_get_nprocs();
    int total_cnt = nprocs * TOP_RANKING;

    // Sort the received rankings based on distance
    qsort((void*)received_rank, total_cnt, sizeof(struct Rank), compare_rank);

    // Measure execution time of the distance-calculating task
    double end_time = MPI_Wtime();

    // Print the final result rankings
    print_result(received_rank);

    // Return execution time of the distance-calculating task
    return end_time;
}

int main(int argc, char* argv[]) {
    // Seed the random
    srand(112525005);

    // Generate random coordinate data
    struct Coord* coord = coord_gen_random_data();

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Store the current rank and the number of processors
    int myrank, nprocs;

    // Get the rank and size of the MPI communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Create MPI datatype for struct Rank
    MPI_Datatype mpi_rank_type = mpi_create_rank_type();

    // Start timing
    double start_time, end_time;
    if(myrank == 0) {
        start_time = MPI_Wtime();
    }

    // Check if the number of processes is within a valid range
    if(nprocs <= 0 || nprocs > MAX_PROCESS_NUM) {
        printf("Invaild process num: %d\n", nprocs);
        exit(EXIT_FAILURE);
    }

    // Check if data can be evenly split among processes
    if(N % nprocs != 0) {
        printf("Couldn't split data evenly!");
        exit(EXIT_FAILURE);
    }

    // Debug print the generated coordinate data
    #ifdef DEBUG
    debug_print_coord_info(coord);
    #endif

    // Calculate distances and rankings for the current process
    struct Rank* rank = calc_all_distance_and_ranking(nprocs, myrank, coord);

    // Gather ranks from all processes to root
    struct Rank* received_rank = malloc(nprocs * TOP_RANKING * sizeof(struct Rank));
    MPI_Gather(rank, TOP_RANKING, mpi_rank_type, 
                        received_rank, TOP_RANKING, mpi_rank_type, 0, MPI_COMM_WORLD);

    // Perform post-processing and print results
    if(myrank == 0) {
        end_time = post_process_and_print_result(received_rank);
        printf("Time elapsed: %.2fs.\n", end_time - start_time);
    }

    // Free dynamically allocated memory
    free(coord);
    free(rank);
    free(received_rank);

    // Free MPI datatype and finalize MPI
    MPI_Type_free(&mpi_rank_type);
    MPI_Finalize();

    // Exit program
    return EXIT_SUCCESS;
}
