from dolfin import *
from ufl import tanh
import numpy as np
import csv
import os

dt = 2*pi*0.01
t = 0
numPeriods = 20
tEnd = 2*pi*numPeriods
saveInterval = 1

Ehigh = 0.0030
Elow = 0.0001
ep = 0.1

dtMin = 2*pi*0.000000001
dtMax = 2*pi*0.01

mesh = IntervalMesh(100,0,1)

# opts = {'newton_solver': {'relative_tolerance': 1e-6}}
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

counter = 0
set_log_active(False)

elY = FiniteElement("Lagrange", mesh.ufl_cell(),1)
elZ = FiniteElement("Lagrange", mesh.ufl_cell(),1)
el = MixedElement([elY, elZ])
V = FunctionSpace(mesh, el)

def boundary_L(x, on_boundary):
    return on_boundary and near(x[0],0)

def boundary_R(x, on_boundary):
    return on_boundary and near(x[0],1)

pinBC = DirichletBC(V.sub(0), Constant(0.0), boundary_L)
noBendingBC = DirichletBC(V.sub(1), Constant(0.0), boundary_R)
bcs = [pinBC, noBendingBC]

# The Neumann BC on the left enters into the variational formulation. We'll use
# a mesh function to mark the LHS of the domain.
leftBoundary = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
leftBoundary.set_all(0) # initialize the function to zero

# create a SubDomain subclass, specifying the portion of the boundary with x[0] < 1/2
class Lefthalf(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0)

# use this Lefthalf object to set values of the mesh function to 1 in the subdomain.
Lefthalf().mark(leftBoundary, 1)
# Define a new measure dsL that just integrates over the left boundary.
dsL = Measure("ds", domain=mesh, subdomain_data=leftBoundary)

# Define variational problem

# u*(yt + (Ez)'') = 0
# v*(z - y'') = 0.
# z(1) = 0 = z'(1)
# y(0) = 0, y'(0) = f(t)
# for trial functions u,v st u(0) = 0 and v(1) = 0. Here, E = E(z), so (E(z))' = E'*z'.

# IBPs
# int(u*yt - u'(Ez)') + [u(Ez)'] = 0. The boundary terms vanish here, as u(0) = 0 and z'(1) = 0 (which makes E' vanish too).
# int(v*z + v'y') - [vy'] = 0. The boundary terms become [] = - v*f(t), as v(1) = 0.

# So, we have the variational formulation
# int(u*yt - u'(Ez)') = 0
# int(v*z + v'y') + v(0)*f(t) = 0

# Using implicit Euler gives replaces all y,z with y,z at the next timepoint, giving
# int[u*(y - yPrev) - dt*u'(Ez)'] = 0
# int[v*z + v'y'] + v(0)*f(t) = 0

# We don't need to multiply the second equation by dt, but doing so seems consistent:
# int[u*(y - yPrev) - dt*u'(Ez)'] = 0
# dt*int[v*z + v'y'] + dt*v(0)*f(t) = 0

# Define solution and test functions. sol = [u,v].
sol = Function(V)
solPrev = Function(V)
dsol = TrialFunction(V)
test = TestFunction(V)

# Define the value taken by f(t) as a Constant so that we can compile the
# residual once for fast solution.
bcVal = Constant(0.0)

# Do the same for dt.
dtVal = Constant(dt)

# Define the initial condition.
solPrev.interpolate(Constant((0.0,0.0)))
sols = solPrev.vector()[:]
ts = [t];

def f(t):
    return 0.1*sin(t)

def E(z):
    return Elow + (Ehigh - Elow)*(0.5 + 0.5*tanh(z/ep))

def residual(sol,solPrev):
    # Access the components of sol and test.
    (y, z) = split(sol)
    (yPrev, zPrev) = split(solPrev)
    (u, v) = split(test)
    # Add up the contributions from each equation.
    I1 = (u*(y - yPrev) - dtVal*dot(grad(u),grad(E(z)*z)))*dx
    I2 = dtVal*(v*z + dot(grad(v),grad(y)))*dx
    BC = dtVal*(v*bcVal)*dsL
    return I1 + I2 + BC

# Loop over time.
while t < tEnd and dt >= dtMin:
    counter += 1

    # Increment time.
    t += dt

    # Assign the value of the driving BC.
    bcVal.assign(f(t))

    F = residual(sol, solPrev)
    J = derivative(F, sol, dsol)
    problem = NonlinearVariationalProblem(F, sol, bcs, J, form_compiler_parameters = ffc_options)
    solver = NonlinearVariationalSolver(problem)

    # Solve for the values at the next timestep.
    try:
        numIter, converged = solver.solve();
    except:
        converged = False

    # If we didn't converge, try again from the old solution with a smaller dt.
    if not converged:
        # Revert to the previous state.
        assign(sol, solPrev)
        t -= dt
        counter -= 1
        dt *= 0.5
        dtVal.assign(dt)
        if dt < dtMin:
            print("Minimum dt reached")
        continue

    # If we converged in only a few iterations, try doubling the timestep next time.
    if numIter < 5:
        dt = min(2*dt, dtMax)
        dtVal.assign(dt)

    print((t,dt))
    
    # Store the solution.
    if (counter % saveInterval == 0):
        sols = np.column_stack((sols, sol.vector()[:]))
        ts.append(t)

    # Update solPrev with the new solution.
    assign(solPrev, sol)


# Saving solution
directory = './'+'Elow='+str(Elow)+'_'+'Ehigh='+str(Ehigh)+'_'+'ep='+str(ep)+'/'
print(directory)
if not os.path.exists(directory):
    os.mkdir(directory)
with open(directory + 'y.csv', 'w') as csvfile:
         csv.writer(csvfile, delimiter=' ').writerows(np.flipud(sols[V.sub(0).dofmap().dofs()[:],:]))
with open(directory + 'z.csv', 'w') as csvfile:
         csv.writer(csvfile, delimiter=' ').writerows(np.flipud(sols[V.sub(1).dofmap().dofs()[:],:]))
with open(directory + 't.csv', 'w') as csvfile:
        csv.writer(csvfile, delimiter=' ').writerow(ts)