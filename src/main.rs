use rand::Rng;
use rayon::prelude::*;

const POPULATION_SIZE: usize = 10000;
const GENERATIONS: usize = 10000;
const MUTATION_RATE: f64 = 0.05;
const MIN_PROGRAM_LENGTH: usize = 6;
const MAX_PROGRAM_LENGTH: usize = 64;
const MEMORY_SIZE: i32 = 256;
const MAX_EXECUTION_STEPS: usize = 1000;

#[derive(Clone)]
struct SUBLEQProgram {
    code: Vec<i32>,
}

impl SUBLEQProgram {
    fn new(length: usize) -> Self {
        let mut rng = rand::thread_rng();
        SUBLEQProgram {
            code: (0..length).map(|_| rng.gen_range(-MEMORY_SIZE..MEMORY_SIZE)).collect(),
        }
    }

    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        for gene in &mut self.code {
            if rng.gen::<f64>() < MUTATION_RATE {
                *gene = rng.gen_range(-MEMORY_SIZE..MEMORY_SIZE);
            }
        }
    }

    fn execute(&self) -> (Vec<i32>, usize) {
        let mut memory = vec![0; MEMORY_SIZE as usize];
        memory[..self.code.len()].copy_from_slice(&self.code);
        
        let mut pc = 0;
        let mut steps = 0;
        while pc < memory.len() - 2 && steps < MAX_EXECUTION_STEPS {
            let a = memory[pc].rem_euclid(MEMORY_SIZE as i32) as usize;
            let b = memory[pc + 1].rem_euclid(MEMORY_SIZE as i32) as usize;
            let c = memory[pc + 2].rem_euclid(MEMORY_SIZE as i32) as usize;
            
            if a < memory.len() && b < memory.len() {
                memory[a] = memory[a].wrapping_sub(memory[b]);
                if memory[a] <= 0 {
                    pc = c % memory.len();
                } else {
                    pc += 3;
                }
            } else {
                pc += 3;
            }
            steps += 1;
        }
        (memory, steps)
    }

    fn fitness(&self) -> usize {
        let (result, steps) = self.execute();
        let mut max_copies = 0;
        for i in 0..result.len() - self.code.len() {
            let mut correct = 0;
            for (j, &val) in self.code.iter().enumerate() {
                if result[i + j] == val {
                    correct += 1;
                } else {
                    break;
                }
            }
            max_copies = max_copies.max(correct);
        }
        // Reward full copies, shorter programs, and fewer execution steps
        if max_copies == self.code.len() {
            max_copies * 1000 / (self.code.len() * steps.max(1))
        } else {
            max_copies
        }
    }

    fn verify_replication(&self) -> bool {
        let (result, _) = self.execute();
        for i in self.code.len()..result.len() - self.code.len() {
            if self.code == result[i..i + self.code.len()] {
                return true;
            }
        }
        false
    }
}

fn crossover(a: &SUBLEQProgram, b: &SUBLEQProgram) -> SUBLEQProgram {
    let mut rng = rand::thread_rng();
    let min_len = a.code.len().min(b.code.len());
    let max_len = a.code.len().max(b.code.len());
    let split = rng.gen_range(0..min_len);
    let child_len = rng.gen_range(min_len..=max_len);

    let mut child = SUBLEQProgram { code: vec![0; child_len] };
    
    // Copy from first parent up to split point
    child.code[..split].copy_from_slice(&a.code[..split]);
    
    // Copy from second parent after split point, up to the minimum length
    let remaining = min_len - split;
    child.code[split..min_len].copy_from_slice(&b.code[split..min_len]);
    
    // If child is longer than min_len, fill the rest randomly
    if child_len > min_len {
        for i in min_len..child_len {
            child.code[i] = rng.gen_range(-MEMORY_SIZE..MEMORY_SIZE);
        }
    }

    child
}

fn main() {
    let mut rng = rand::thread_rng();
    let mut population: Vec<SUBLEQProgram> = (0..POPULATION_SIZE)
        .map(|_| SUBLEQProgram::new(rng.gen_range(MIN_PROGRAM_LENGTH..=MAX_PROGRAM_LENGTH)))
        .collect();

    for generation in 0..GENERATIONS {
        let fitness_scores: Vec<usize> = population.par_iter().map(|p| p.fitness()).collect();
        
        let best_fitness = *fitness_scores.iter().max().unwrap();
        let best_program = &population[fitness_scores.iter().position(|&r| r == best_fitness).unwrap()];
        
        println!("Generation {}: Best fitness = {}", generation, best_fitness);
        if best_program.verify_replication() {
            println!("Self-replicator found: {:?}", best_program.code);
            let (result, steps) = best_program.execute();
            println!("Execution result: {:?}", result);
            println!("Steps taken: {}", steps);
            return;
        }

        let mut new_population = Vec::with_capacity(POPULATION_SIZE);
        
        while new_population.len() < POPULATION_SIZE {
            let parent1 = &population[rng.gen_range(0..POPULATION_SIZE)];
            let parent2 = &population[rng.gen_range(0..POPULATION_SIZE)];
            let mut child = crossover(parent1, parent2);
            child.mutate();
            new_population.push(child);
        }

        population = new_population;
    }

    println!("No perfect self-replicator found within {} generations", GENERATIONS);
}
