import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy 

parser = argparse.ArgumentParser(
    description = "Classical Dynamics simulation for a quantum system"
)

parser.add_argument("Integrator", help = """different numerical integrators used to update position and momentum:
euler  --  Euler step
rk2    --  Runge Kutta second order
rk4    --  Runge Kutta fourth order
ark4   --  adaptive step size runge kutta method
""",type = str, default = 'euler')

parser.add_argument(
    "-n","--n", help = "Number of particles in simulation", type = int, default = 3
)

parser.add_argument(
    "-w","--width",help = "width of simulation box in arbitrary dimension", type = int, default = 10
)

parser.add_argument(
    "-pot","--potential",help = """potential used for simulation
    grav  --  gravitational potential
    spring --  linear spring potential  
    """, type = str, default = 'gravity'
)

parser.add_argument(
    "-dim","--dimension",help = "Dimension of simulation (plotting only works in 2 dimensions for now)", type = int, default = 2
)

parser.add_argument(
    "-s","--speed",help = "Initial speed of particles", type = int, default = 1
)

parser.add_argument(
    "-t","--runtime",help = "total run time of simulation in seconds", type = int, default = 10
)

parser.add_argument(
    "-ndt","--ndt",help = "number of time steps per second, total time steps would thus be ndt*t", type = int, default = 1000
)

parser.add_argument(
    "-tol","--tolerance",help = "collision tolerance between particles", type = float, default = 10**(-1)
)

parser.add_argument(
    "-grad","--gradientstep", help = "gradient step", type = float, default = 10**(-3)
)

parser.add_argument(
    "-pol","--polarity", help = "percentage of spin 1/2 particles in the system", type = float, default = 1.0
)

args = parser.parse_args()



class particle: #sets up particle class with momentum and position arrays
    def __init__(self,position,momentum,spin):
        self.position = position
        self.momentum = momentum
        self.dimension = len(position)
        self.spin = spin
        
        
def gravity(r): #defines gravitational potential
    return 1/r

def spring(r):
    return -0.5*r**2

def hamiltonian(particles,pot):#calculates the total energy of the system at a given time step
    kinetic_energy = 0
    Average_KE = 0
    potential = 0
   
    for i,particle in enumerate(particles):
        kinetic_energy += 0.5*np.linalg.norm(particle.momentum)**2
        for j,partner in enumerate(particles):
            if partner != particle:

                #here is where we could add a tolerance so the particles don't get infinite force when close
                sep_vec = particle.position - partner.position
                potential += -pot(np.linalg.norm(sep_vec))


    Average_KE = kinetic_energy/len(particles)
    potential = potential/2


    Hamiltonian = kinetic_energy + potential
    return (kinetic_energy, Average_KE,potential,Hamiltonian)


def gradient(sep_vec,pot,step):    #takSes in seperation vector between two particles returns force array in arbitrary dimension
    Force = np.zeros_like(sep_vec)

    for i in range(len(sep_vec)):
        q1 = np.copy(sep_vec)
        q2 = np.copy(sep_vec)
        q1[i] += -step
        q2[i] +=  step
        v2 = pot(np.linalg.norm(q2))
        v1 = pot(np.linalg.norm(q1))
        Force[i] = (v2-v1)/(2*step)
        
    return (Force)

def initialize(N,Dim,width,speed,polarity):#initializes the position and momentum of particles
    particles = []
    A = int(polarity *N)
    B = N - A
    print(A,B)

    
    for i in range(N):
        position = width*(2*np.random.rand(Dim) - 1) #uniformly distributes in particles throughout the box
        momentum = speed*(2*np.random.rand(Dim) - 1)#uniformly distrubutes momentum of particles up to speed parameter
        spin = 0
        if polarity == 1:
            spin = 0
        
        else: 
            if i+1<= A:
                spin = 0.5
            else:
                spin = -0.5
        particles.append(particle(position,momentum,spin))
    return particles

def update(particles,dt,pot,step,tol,width): #updates the position and momentum at a given time step
    # new_position = np.zeros(len(particles[0].position))
    # new_momentum = np.zeros(len(particles[0].momentum))


    N = len(particles)
    Dim =  len(particles[0].position)

    total_momentum = np.zeros((N,N,Dim))


    for i,particle in enumerate(particles):
        
        new_position = np.zeros((N,Dim))
        momentum_change = np.zeros_like(particle.momentum)

        #particle elastically bounces off walls
        for q,cord in enumerate(particle.position): 
            if abs(cord) >= width:
                particle.momentum[q] = -particle.momentum[q]


        for j,partner in enumerate(particles):
            if partner != particle:
                if particle.spin == 0:

                    #here is where we could add a tolerance so the particles don't get infinite force when close
                    sep_vec = particle.position - partner.position
                    
                    # if particles are a suitable distance from one another continue as normal
                    if np.linalg.norm(sep_vec) > tol:
                        force = gradient(sep_vec,pot,step)
                        momentum_change = force*dt
                        
                    #if particles are too close make them bounce off one another
                    if np.linalg.norm(sep_vec) <= tol:
                        # new_momentum = np.linalg.norm(particle.momentum)*unit_sep_vec
                        momentum_change = -2*particle.momentum

                    
                if abs(particle.spin) == 0.5:
                    if partner.spin == -particle.spin:
                        sep_vec = particle.position - partner.position
                    
                        # if particles are a suitable distance from one another continue as normal
                        if np.linalg.norm(sep_vec) > tol:
                            force = gradient(sep_vec,pot,step)
                            momentum_change = force*dt
                            
                        #if particles are too close make them bounce off one another
                        if np.linalg.norm(sep_vec) <= tol:
                            # new_momentum = np.linalg.norm(particle.momentum)*unit_sep_vec
                            momentum_change = -2*particle.momentum

                total_momentum[i,j,:] = momentum_change


    for i,particle in enumerate(particles):
        particle.momentum += np.sum(total_momentum[i,:,:],axis=0)
        particle.position += particle.momentum*dt
        
    return particles

#N             -- number of particles
#Dim           -- dimension of simulation (plotting only works in 2 for now)
#t             -- total run time of simulation
#Ndt           -- number of time steps per second
#pot           -- potential to be used in simulation
#gradientstep  -- the small differential step used to calculate the gradient, smaller is betterr
#width         -- width of box particles are constrained to
#speed         -- maximum initial speed of particles 
#tol           -- the minimum distance between particles before they elastically collide.

def main(**kwargs): 

    N = args.n
    Dim = args.dimension
    t = args.runtime
    Ndt = args.ndt
    width = args.width
    speed = args.speed
    tol = args.tolerance 
    gradientstep = args.gradientstep
    polarity = args.polarity


    pot = gravity
    # if args.potential == 'grav' or args.potential == 'gravity':
    #     pot = gravity
    # elif args.potential == 'spr' or args.potential == 'spring':
    #     pot = spring
    # else:
    #     raise Exception("Sorry, not a valid potential")
    
    
    particles = initialize(N,Dim,width,speed,polarity)
    

    
    
    dt = 1/Ndt
    data = np.zeros([N,Dim,Ndt*t])
    hamil = np.zeros((int((Ndt*t)/10+1),3))
     
    #core update loop
    for k in range(1,Ndt*t+1):
        particles = update(particles,dt,pot,gradientstep,tol,width)
        if k%10 == 0: 
            hamil[int(k/10),0] = hamiltonian(particles,pot)[0]
            hamil[int(k/10),1] = hamiltonian(particles,pot)[3]
            hamil[int(k/10),2] = hamiltonian(particles,pot)[2]
        for i,particle in enumerate(particles):
            data[i,:,k-1] = particle.position
    
    

    initial_potential = hamil[0,2]
    initial_KE = hamil[0,0]
    fig, ax = plt.subplots(1,2, figsize = (15,15))
    ax[1].plot(hamil[1:,0]-initial_KE,label='Kinetic Energy')
    ax[1].plot(hamil[1:,2]-initial_potential,label='Potential energy')
    ax[1].plot(hamil[1:,1]-initial_potential-initial_KE,label='Total energy')
    
    # ax[1].plot((hamil[1:,2]-initial_potential)/(hamil[1:,0]-initial_KE))
    #plotting the trajectories
    for i in range(N):
        
        ax[0].plot(data[i,0,:],data[i,1,:])
    plt.legend()
    plt.show()

    #Idk how to plot this, particles is a list of objects, we need to pass in an array of x at all t and an array of 
    #y values at all t in order to plot. That means we need to exctract the position data from the objects in particles
    #list, but this is hard!

    return 


main()
    