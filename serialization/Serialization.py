import numpy as np
import pandas as pd
from PrimitiveFunctions import spherical_to_cartesian,cartesian_to_spherical,rot_matrix
from scipy.interpolate import RegularGridInterpolator,CubicSpline,LinearNDInterpolator

DEFAULT_ANGLE_GUESS=np.radians(30)
DISTANCE_MAGNITUDE=1e6
ANGLE_MAGNITUDE=np.radians(30/3600)
ITERATION_LIMIT=0.5
SAMPLE_FRACTION=1/3

# WARNING: BLACK BOX CODE
# The following code has been idealized as a black box that prioritizes speed over
# readability. If you are debugging your code and ended up here, I highly recommend
# exhausting all other options before looking for bugs here. In any case, comments 
# will be left very abundantly so that you can follow my reasoning as I wrote this. 

# Serialization functions for the creation of the geometry files

# Unfortunately, the same standard of elegance used for the rest of the 
# code cannot really be applied here, so be prepared for long functions 
# and "choices". The good part is that the average user is not really 
# meant to see this code, and the only the classes in Primitives have
# to really deal with this. The objective here is to create a way for 
# geometry to be saved to file while saving time, memory and storage, 
# as well as giving Plotly an easier time afterwards.

############################################################################################

# Creates a grid to be saved in our binary file and to be posteriorly 
# used by the event viewer to reconstruct the simulation space. Takes a 
# resolution in arc seconds for calculating the grid.

# These grids can get very large, so it is better if we parallelize the creation of 
# the grids themselves. Creating a version of this "rendering" algorithm that uses
# the GPU might also not be a bad idea.

def distance_array(guess):
    guess0=guess
    guess_array=[guess0]

    while guess>guess0/DISTANCE_MAGNITUDE:
        guess_array.append(guess/10)
        guess=guess/10
    return np.array(guess_array)

def angle_array(guess):
    guess_array=[guess]
    while guess>ANGLE_MAGNITUDE:
        guess_array.append(guess/2)
        guess=guess/2
    return np.array(guess_array)
        
def grid_2D(quantity,coords,pack=True):

    grid=[]
    coords1,coords2=coords

    for coord1 in coords1:
        for coord2 in coords2:

            if not callable(quantity):
                grid.append(np.array([coord1,coord2,quantity]))
            else:
                if pack:
                    grid.append(np.array([coord1,coord2,quantity((coord1,coord2))]))
                else:
                    grid.append(np.array([coord1,coord2,quantity(coord1,coord2)]))

    return np.array(grid)

def refined_array(refine_coord,old_step,new_step,coord_system="Cartesian"):

    coords_temp=[]

    for i in range(len(refine_coord[:-1])):

        buffer=np.linspace(refine_coord[i],refine_coord[i]+old_step,
                                int(np.ceil(old_step/new_step)+1))

        coords_temp.append( buffer[1:-1])
        actual=buffer[1]-buffer[0]
        
    return coords_temp,actual
            
def refine_2D(function,old_grid,shape,refine_coord,new_step,old_step,pack,coord_system):

    coords_temp,actual=refined_array(refine_coord,old_step,new_step,coord_system)
        
    old_grid=old_grid.reshape(shape[0],shape[1],3)
    new_grid=[]
    new_coords2=[]

    for row in old_grid:
        new_row=[]
        for i in range(len(row[:-1])):
            
            coord1,coord2,value=row[i]
            new_row.append(row[i])

            if not np.isin(coord2,new_coords2):
                new_coords2.append(coord2)

            new_row.extend(grid_2D(function,[[coord1],coords_temp[i]],pack))

            if not np.all(np.isin(coords_temp[i], new_coords2)):
                new_coords2.extend(coords_temp[i])

            if i==len(row[:-1])-1:
                coord1,coord2,value=row[-1]
                new_row.append(row[-1])
                if coord2 not in new_coords2:
                        new_coords2.append(coord2)

        new_grid.append(new_row)

    new_grid=np.array(new_grid)
    new_grid=new_grid.reshape(-1,shape[2])

    new_shape=np.copy(shape)
    new_shape[1]=len(new_coords2)

    new_coords2=np.array(new_coords2)
    return new_grid,new_coords2,actual,new_shape

def refine_3D(position,function,old_grid,shape,refine_coord,
              new_step,old_step,
              rotation_history,pack,
              coord_system,func_coord_system,
              pos_dependent):

    coords_temp,actual=refined_array(refine_coord,old_step,new_step,coord_system)
        
    old_grid=old_grid.reshape(shape[0],shape[1],shape[2],shape[3])
    new_grid=[]
    new_coords3=[]

    for row in old_grid:

        new_row=[]

        for vec in row:

            new_vec=[]

            for i in range(len(vec[:-1])):
                
                coord1,coord2,coord3,value=vec[i]

                if coord_system=="Spherical":
                    coord1,coord2,_=cartesian_to_spherical(coord1,coord2,coord3)
                    coord3=refine_coord[i]
                new_vec.append(vec[i])

                if not np.isin(coord3,new_coords3):
                    new_coords3.append(coord3)

                new_vec.extend(grid_3D( position,function,
                                       [[coord1],[coord2],coords_temp[i]],pos_dependent,
                                       rotation_history,func_coord_system,
                                       pack,coord_system))

                if not np.all(np.isin(coords_temp[i], new_coords3)):
                    new_coords3.extend(coords_temp[i])

                if i==len(vec[:-1])-1:
                    new_vec.append(vec[-1])
                    coord1,coord2,coord3,value=vec[-1]
                    coord3=refine_coord[-1]
                    if coord3 not in new_coords3:
                        new_coords3.append(coord3)

            new_row.append(new_vec)
        new_grid.append(new_row)

    new_grid=np.array(new_grid)
    new_grid=new_grid.reshape(-1,shape[3])

    new_shape=np.copy(shape)
    new_shape[2]=len(new_coords3)
    new_coords3=np.array(new_coords3)

    return new_grid,new_coords3,actual,new_shape

def find_best_guess(prediction,threshold,guesses):
    
    mask=np.array(prediction)<=threshold
    prediction=prediction[mask]
    where=np.argmin(abs(prediction-threshold))

    candidates=guesses[mask]
    best_guess=candidates[where]

    return best_guess

# Checking the trend is a way to cut short the optimization if we are not likely to get 
# the result we want. This is not a foolproof method, but the worst case scenario just 
# causes a finer resolution than needed 

def check_trend2D(old_guesses,percentual,new_guesses,threshold):

    order=np.argsort(old_guesses)
    trend=CubicSpline(np.array(old_guesses)[order],np.array(percentual)[order])
    predict=trend(new_guesses)

    if np.any(predict<=threshold):

        return True,find_best_guess(predict,threshold,new_guesses)

    else:

        return False,None

def check_trend3D(old_guesses,percentual,new_guesses,threshold):

    old_guesses=np.array(old_guesses)

    x=old_guesses[:,0]
    y=old_guesses[:,1]
    x_idxs=np.sort(np.unique(x,return_index=True)[1])
    y_idxs=np.sort(np.unique(y,return_index=True)[1])
    try:
        predictor=RegularGridInterpolator((x[x_idxs],y[y_idxs]),
                                            percentual,method="linear",bounds_error=False,
                                            fill_value=None)
    except ValueError:
        predictor=LinearNDInterpolator(old_guesses,
                                        percentual,
                                        fill_value=None)     
                    
    outer_grid, inner_grid = np.meshgrid(
        new_guesses[0], new_guesses[1], indexing="ij"
    )

    predictor_coords=np.column_stack((outer_grid.ravel(),
                                        inner_grid.ravel()))
    
    prediction=predictor(predictor_coords)


    if np.any(prediction<=threshold):

        return True,find_best_guess(prediction,
                                    threshold,
                                    predictor_coords)
    else:
        return False,None
    
def coord_arrays2D(coord_lims,guesses,coord_system):

    if coord_system=="Spherical":

        coords1=np.arccos(np.linspace(-1,1,int(np.ceil(2/guesses[0])+1)))

        coords2=np.linspace(0,np.nextafter(2*np.pi,0),int(np.ceil(2*np.pi/guesses[1])+1))
    else:

        coords1=np.linspace(coord_lims[0][0],coord_lims[0][1],
                            int(np.ceil((coord_lims[0][1]-coord_lims[0][0])/guesses[0]+1)))
        
        coords2=np.linspace(coord_lims[1][0],coord_lims[1][1],
                            int(np.ceil((coord_lims[1][1]-coord_lims[1][0])/guesses[1]+1)))

    return coords1,coords2

def test_coords2D(coords,coord_system,samples=None,sample_fraction=None):

    coords1,coords2=coords

    if coord_system=="Spherical":
        middle1=np.cos(coords1[:-1])+(np.cos(coords1[1])-np.cos(coords1[0]))/2
        middle1=np.arccos(middle1)
    else:
        middle1=coords1[:-1]+(coords1[1]-coords1[0])/2

    middle2=coords2[:-1]+(coords2[1]-coords2[0])/2

    # Samples random points and checks whether the interpolation produces at 
    # those point interpolated predictions that match the value of the actual
    # function within a given tolerance.

    if samples is None:
        if sample_fraction is None:
            raise ValueError("The user must provide a way for the" \
            " number of samples to be estimated")
        
        sample1=max(1,int(len(middle1)*sample_fraction))
        sample2=max(1,int(len(middle2)*sample_fraction))

    else:

        sample1=samples[0]
        sample2=samples[1]

    points1=np.random.choice(middle1,sample1,replace=False)
    points2=np.random.choice(middle2,sample2,replace=False)

    sample1_temp,sample2_temp=np.meshgrid(points1,points2,indexing='ij')
    test_coords=np.column_stack((sample1_temp.ravel(),sample2_temp.ravel()))

    return test_coords

def test_coords3D(position,coords,coord_system,samples=None,
                  sample_fraction=None):

    coords1,coords2,coords3=coords

    middle1=coords1[:-1]+(coords1[1]-coords1[0])/2

    if coord_system=="Spherical":
        middle2=np.cos(coords2[:-1])+(np.cos(coords2[1])-np.cos(coords2[0]))/2
        middle2=np.arccos(middle2)
    else:
        middle2=coords2[:-1]+(coords2[1]-coords2[0])/2

    middle3=coords3[:-1]+(coords3[1]-coords3[0])/2

    if samples is None:
        if sample_fraction is None:
            raise ValueError("The user must provide a way for the" \
            " number of samples to be estimated")
        
        sample1=max(1,int(len(middle1)*sample_fraction))
        sample2=max(1,int(len(middle2)*sample_fraction))
        sample3=max(1,int(len(middle3)*sample_fraction))

    else:

        sample1=samples[0]
        sample2=samples[1]
        sample3=samples[2]

    points1=np.random.choice(middle1,sample1,replace=False)
    points2=np.random.choice(middle2,sample2,replace=False)
    points3=np.random.choice(middle3,sample3,replace=False)

    sample1_temp,sample2_temp,sample3_temp=np.meshgrid(points1,points2,points3,indexing='ij')

    if coord_system=="Spherical":
        test_coords_cartesian=[]
        test_coords_spherical=np.column_stack((sample1_temp.ravel(),
                                    sample2_temp.ravel(),
                                    sample3_temp.ravel()))
        for coords in test_coords_spherical:
            r,theta,phi=coords
            test_coords_cartesian.append(np.array(spherical_to_cartesian(r,theta,phi)))

    else:
        test_coords_spherical=[]
        test_coords_cartesian=np.column_stack((sample1_temp.ravel(),
                                    sample2_temp.ravel(),
                                    sample3_temp.ravel()))
        for coords in test_coords_cartesian:
            x,y,z=coords
            test_coords_spherical.append(np.array(cartesian_to_spherical(x,y,z)))
    
    return np.array(test_coords_spherical),np.array(test_coords_cartesian)

def test_coords3DSphere(ang_coords,ang_interpolators,r_coords,samples=None,sample_fraction=None):

    if samples is None and sample_fraction is None:
        raise RuntimeError("User must provide a way for the algorithm " \
        "to estimate the number of samples")

    theta,phi=ang_coords
    iradius_interpolator,oradius_interpolator=ang_interpolators

    middle_theta=np.arccos(np.cos(theta[:-1])+(np.cos(theta[1])-np.cos(theta[0]))/2)
    middle_phi=phi[:-1]+(phi[1]-phi[0])/2

    if samples is None:

        sample1=max(1,int(len(np.array(r_coords).flatten())*sample_fraction))
        sample2=max(1,int(len(theta)*sample_fraction))
        sample3=max(1,int(len(phi)*sample_fraction))

    else:

        sample1=samples[0]
        sample2=samples[1]
        sample3=samples[2]

    theta_test=np.random.choice(middle_theta,sample2)
    phi_test=np.random.choice(middle_phi,sample3)

    thetas,phis=np.meshgrid(theta_test,phi_test,indexing='ij')

    ang_test=np.column_stack((thetas.ravel(),phis.ravel()))

    inner_radii=iradius_interpolator(ang_test)
    outer_radii=oradius_interpolator(ang_test)

    # using the middle of the r bins for the random draws is not really possible because 
    # of the fact we do not have the knowledge of how the radius changes with the angular
    # coordinates.

    pairs=list(zip(inner_radii,outer_radii))
    test_coords_spherical=[]
    test_coords_cartesian=[]

    for i in range(len(ang_test)):
        for j in range(sample1):
            r=np.random.uniform(pairs[i][0],pairs[i][1])
            temp=[r]
            temp.extend(ang_test[i])
            test_coords_spherical.append(temp)

    for coords in test_coords_spherical:
        r,theta,phi=coords
        test_coords_cartesian.append(np.array(spherical_to_cartesian(r,theta,phi)))

    return np.array(test_coords_spherical),np.array(test_coords_cartesian)

def comparison_coords3D(position,coords,
                        coord_system,func_coord_system,
                        samples,sample_fraction,
                        rotation_history,
                        pos_dependent):

    coords1,coords2,coords3=coords
    test_coords_spherical,test_coords_cartesian=test_coords3D(position,[coords1,coords2,coords3],
                                coord_system,samples,sample_fraction)

    if coord_system=="Spherical":
        interp_coords=test_coords_spherical
    else:
        interp_coords=test_coords_cartesian

    func_coords=test_coords_cartesian

    # Blocks are a nightmare of their own for serialization because they may be rotated, 
    # and their rotation brings aconsequences a bit harder to deal with than just a phase
    # addition. This is part of why we save the position and orientation f the object separately,
    # and serialize them as if they are at (0,0,0) with no changes in orientation.
       
    if rotation_history is not None:
        for axis,theta in reversed(rotation_history):
            func_coords=func_coords@rot_matrix(axis,-theta).T

    if pos_dependent:
        func_coords=func_coords+position

    if func_coord_system=="Spherical":
        new_func_coords=[]
        for coord in func_coords:
            x,y,z=coord
            new_func_coords.append(cartesian_to_spherical(x,y,z))
        func_coords=new_func_coords

    return np.array(interp_coords),np.array(func_coords)

def comparison_coords3DSphere(position,
                              ang_coords,ang_interpolators,
                              r_coords,
                              samples,sample_fraction,
                              coord_system,func_coord_system,
                              pos_dependent):

    test_coords_spherical,test_coords_cartesian=test_coords3DSphere(ang_coords,ang_interpolators,
                                                                    r_coords,samples,sample_fraction)
    interp_coords=test_coords_spherical


    func_coords=test_coords_cartesian

    if pos_dependent:
        func_coords=func_coords+position

    if func_coord_system=="Spherical":
        new_func_coords=[]
        for coord in func_coords:
            x,y,z=coord
            new_func_coords.append(cartesian_to_spherical(x,y,z))
        func_coords=new_func_coords

    return np.array(interp_coords),np.array(func_coords)

def grid_3D(position,function,coords,pos_dependent,rotation_history,
            func_coord_system,pack=True,
            coord_system="Cartesian"):

    grid=[]
    coords1,coords2,coords3=coords

    for coord1 in coords1:

        for coord2 in coords2:

            for coord3 in coords3:

                if coord_system=="Spherical":

                    x,y,z=spherical_to_cartesian(coord1,coord2,coord3)

                else:

                    x,y,z=coord1,coord2,coord3

                x_temp,y_temp,z_temp=x,y,z
                if rotation_history is not None:
                    pos=np.array([x_temp,y_temp,z_temp])

                    for axis,theta in reversed(rotation_history):
                        pos=pos@rot_matrix(axis,-theta).T

                    x_temp,y_temp,z_temp=pos

                # the idea is for the grid to be saved using the local coordinates, 
                # allowing us to move the object imported from file more easily.

                if pos_dependent:
                    x_temp,y_temp,z_temp=np.array([x_temp,y_temp,z_temp])+position

                if func_coord_system=="Spherical":
                    x_temp,y_temp,z_temp=cartesian_to_spherical(x_temp,y_temp,z_temp)

                if pack:

                    grid.append(np.array([x,y,z,function((x_temp,y_temp,z_temp))]))

                else:

                    grid.append(np.array([x,y,z,function(x_temp,y_temp,z_temp)]))

    return np.array(grid)

# An absolutely horrendous piece of code that should make it possible to store a grid of 
# densities for a sphere with variable radius and density .

def grid_to_dict(grid):

    shape=grid.shape
    dimension=shape[1]-1

    grid_dict={}

    for point in grid:
        if dimension==3:
            coord1,coord2,coord3,result=point
            grid_dict[(coord1,coord2,coord3)]=result
        else:
            coord1,coord2,result=point
            grid_dict[(coord1,coord2)]=result

    return grid_dict
                        
# These very convoluted functions are meant to take a given function and create a grid for 
# later interpolating while optimizing for the smallest resolution possible to achieve
# precision within a given threshold. This should both prevent the worst of aliasing and 
# save us some time and space in the process of saving the geometry for the event viewer.

# Unfortunately, the guess process cannot be parallelized, as doing so defeats the 
# very purpose of why we are doing this (save time, memory and storage). 
def serialize2D(function,coord_lims,guesses,
                          samples=None,coord_system="Cartesian",
                          threshold=0.001,iteration_limit=0.5,
                          sample_fraction=0.25,
                          pack=True): 
    
    
    # We start with initial guesses, then we half it for angular coordinates and 
    # by an order of magnitude for "lengths" every time the test fails.

    # Elegant? No. But it will avoid us having to provide guesses altogether.
    # The objective is to automate this process so the saving can be done with minimal
    # hassle

    if guesses is not None:
        guess1,guess2=guesses
    else:
        if coord_system=="Cartesian":
            space1=coord_lims[0][1]-coord_lims[0][0]
            space2=coord_lims[1][1]-coord_lims[1][0]
            guess1,guess2=space1/10,space2/10

        else:

            # The size of the sphere arc covered by a given angle is dependent on the 
            # radius of the sphere (ex: A 30 arc second resolution for a sphere the size 
            # of Earth has an arc length of 30 meters), so we need more guesses for the 
            # angular coordinates.

            guess1,guess2=DEFAULT_ANGLE_GUESS,DEFAULT_ANGLE_GUESS

    # We create arrays of guesses to allow for different resolution in different axes, 
    # as the properties of the object may change in ways that benefit from this treatment.
    # A certain degree of guessing has to be applied here, because we do not have a one size
    # fits all standard for how many guesses we will need, and we cannot let these grids 
    # be absurdly sized. A good rule of thumb is to use the precision used for the ETOPO
    # Earth elevation model as the limit for the angular precision (30 arc seconds) and 
    # 6 orders of magnitude for distances. These numbers are a bit arbitrary and the 
    # limits will be left as global variables to allow them to be changed more easily, 
    # but it should be enough.
    # 
    # EXAMPLE: For the case of the Earth, 6 orders of magnitude precision for the radius grid would
    # be 60 meters. That's A LOT of precision.
    # 
    # The worst case scenario for either grid is grim, but unless the software is 
    # genuinely being misused it should never be reached. This is more insurance 
    # than an actual real thing that might happen.   
 
    # We assume SI units are being used, as much as I hate assumptions.


    if coord_system=="Cartesian":
        guesses1 = distance_array(guess1)
        guesses2 = distance_array(guess2)

    else:
        guesses1 = angle_array(guess1)
        guesses2 = angle_array(guess2)

    # Separate resolution guesses per coordinate may optimize the 
    # resolution for some objects depending on how the property to be "gridded"
    # changes

    # One thing we have to consider is when insisting on something is a waste of time.
    # Increasing the precision in one array while the other stays the same until we go to 
    # the upper limit is highly unlikely to achieve our objective if the precision is not
    # showing a trend that is compatible with achieving our objective any time soon.
    # Consider the case of a sphere: how likely is it that a 30 arc second precision in our 
    # phi coordinate will provide us with our desired precision while our theta guess 
    # is still at 10 degrees?

    # Given that our guesses array for distances will have 6 elements, the idea is:
    #   - With one guess fixed, we calculate up to 4 of the next grids, and store 
    #   the resulting percentual precision.
    #   - We then interpolate these 4 precision values and evaluate the trend, 
    #   asking ourselves: Is it likely we will achieve our target precision 
    #   with this fixed guess?
    #   - If the answer is no, we can skip the useless iterations and change our 
    #   outer guess, then continue our process.
    #
    # Is it convoluted? Yes. But in doign so we can save ourselves a couple million 
    # useless calculations in the best case scenario.

    for guess1 in guesses1:

        percentual=[]
        guess=[]
        grid=None
        step_coord=None


        for guess2 in guesses2:

            # We start with a strict limit of 0.5. This means that the retopology attempt
            # will try half of the guesses for the inner loop and check if the threshold
            # is even achievable. Allied with the nested grids, this should do a good job 

            if len(percentual)>int(iteration_limit*len(guesses2)):
                possible,best_guess=check_trend2D(guess,percentual,
                                                  guesses2[(len(percentual)-1):],threshold)
                if possible:
                    guess2=best_guess
                else:
                    continue

            if grid is None:
                coords1,coords2=coord_arrays2D(coord_lims,[guess1,guess2],coord_system)

                actual=coords2[1]-coords2[0]
                step_coord=actual

                shape=[len(coords1),len(coords2),3]
                grid=grid_2D(function,[coords1,coords2],pack)

                    
            else:

                grid,coords2,step_coord,shape=refine_2D(function,grid,shape,coords2,
                                                     guess2,step_coord,pack,coord_system)

            interp_grid=np.copy(grid)
            interp_grid=interp_grid.reshape(shape[0],shape[1],shape[2])

            interp = RegularGridInterpolator((coords1, coords2), 
                                    interp_grid[:,:,2], method="linear",
                                    bounds_error=False,fill_value=None)
            if samples is not None:
                test_coords=test_coords2D([coords1,coords2],coord_system,samples=samples)
            else:
                test_coords=test_coords2D([coords1,coords2],
                                          coord_system,sample_fraction=sample_fraction)

            test_points=interp(test_coords)

            if pack:
                comparison=function((test_coords[:,0],test_coords[:,1]))
            else:
                comparison=function(test_coords[:,0],test_coords[:,1])

            # We compare the mean deviation from the expectation
            # to our percentual threshold. 
            mean=np.mean(np.divide(abs(comparison-test_points),comparison))

            if mean<=threshold:
                return grid
            
            else:
                percentual.append(mean)
                guess.append(guess2)
   
    raise RuntimeError("Requested resolution is not achievable. " \
    "Lower threshold and try again")       
    
# We do not tie the resolution of the 3D case to the 2D case for two reasons:
#   - The Block class won't have a 2D case to use
#   - The property to be "gridded" in 3D may have a weaker dependence on 
#    the variation of the property of interest in 2D
# 
# All of this will result in longer code and more coding time, 
# but will save us computational resources.
 
# At the moment, the serialization of functions depending on the 3 coordinates will only 
# work with density, as it only depends on the position. Dealing with the stopping power 
# would require to deal with its dependence on the density itself. A separate function
# would probably be better.
 
# The idea is: We grid both inner and outer radius of the object (0 for the inner radius
# if we are talking about a Sphere object) and 
# We divide the 3D case in one case for the spherical solids and one for the 
# cuboids. Not exactly the most elegant thing in the world, but the cases are different
# enough that trying to make a one size fits all function is not really possible and 
# Python does not allow to define multiple signatures for the same function name like C.

# The serialization of the 3D sphere with variable surface and density is the worst case 
# scenario and can be expected to run slowly, but it will be rare enough I am confident 
# most users will probably never stumble upon it, as even needing this would require you 
# to have built a very complicated geometry yourself. For most people, not a lot 
# of resolution will be needed.

# Consider an exponential atmosphere, for example: since the SphereShell will have perfectly
# symmetrical density and constant inner and outer radius, you do not need a lot of angular 
# resolution, but rather higher radial resolution, and even then, I estimate probably in the
# scale of kilometers,depending on your thickness.

def serialize3D_sphere(position,inner,outer,function,guesses=None,
                        samples=None,sample_fraction=0.125,
                        threshold=0.001,iteration_limit=0.5,
                        coord_system="Cartesian",
                        func_coord_system="Cartesian",
                        pos_dependent=True,
                        pack=True):
    
    if guesses is not None:
        guess1,guess2=guesses
    else:
        guess1,guess2=DEFAULT_ANGLE_GUESS,DEFAULT_ANGLE_GUESS

    guesses1 = angle_array(guess1)
    guesses2 = angle_array(guess2)

    # Tying together the angular resolution for both inner and outer radius is
    # something that will often result in higher than needed resolution for one 
    # of the surfaces, however we must consider that the alternative would be 
    # 5 NESTED LOOPS. 
    guesses=[10,100,1000,1e4,1e5,1e6]

    for guess1 in guesses1:
        
        percentual=[]
        guess_outer=[]
        oradius_grid=None
        step_coord=None

        iradius_grid=None
        step_coord2=None

        for j in range(len(guesses2)):

            guess2=guesses2[j]
            assigned_outer=False
            # Instead of the simple trend checking structure of the 2D case, for the 
            # 3D case, we have two checks for the trends: one while trying different 
            # guesses for r and one while trying different guesses for phi. The phi case 
            # takes into account a combination of both phi and r guesses, using a 3D interpolator. 
            if j>int(len(guesses2)*iteration_limit):

                outer_guesses=guesses2[j:]
                new_guesses=[outer_guesses,guesses]

                possible,best_guesses=check_trend3D(guess_outer,
                                                    percentual,new_guesses,threshold)

                if possible:
                    assigned_outer=True
                    guess2,guess=best_guesses
                else:
                    continue


            # Gridding both inner and outer radius is necessary even if they are constants
            # because we do not know the dependence of the density on the angular coordinates.
            # If the dependence is minimal, we can expect this to converge to the lowest resolution

            # Both inner and outer radius share the same coordinates, 
            # so we can grid them together

            if oradius_grid is None or iradius_grid is None:
                coords1,coords2=coord_arrays2D(None,[guess1,guess2],coord_system="Spherical")

                actual=coords2[1]-coords2[0]
                step_coord=actual

                shape_angle=[len(coords1),len(coords2),3]

                iradius_grid=grid_2D(inner,[coords1,coords2],pack)
                oradius_grid=grid_2D(outer,[coords1,coords2],pack)

                
            else:
                # It would be faster to not have to redefine the coords2 array and everything,
                # however by re-using the function defined for the 2D case the code gets 
                # significantly simpler to debug, and something this complex needs this 
                # sort of tradeoff

                iradius_grid,new_coords2,new_step,new_shape=refine_2D(inner,iradius_grid,shape_angle,coords2,
                                                            guess2,step_coord,pack,coord_system="Spherical")
                oradius_grid,coords2,step_coord,shape_angle=refine_2D(outer,oradius_grid,shape_angle,coords2,
                                                          guess2,step_coord,pack,coord_system="Spherical")

            # We here define the resolution a little bit different from in the previous case.
            # Here, we define the resolution itself as a percentual resolution, spanning 
            # the same interval of 6 orders of magnitude, and for each pair of theta and 
            # phi coordinates an array with the values of r with the given resolution will be 
            # generated for the creation of the grid. This is the cleanest way to deal with
            # the more complex geometry we expect for a full model of the Earth.
            
            iradius_matrix=iradius_grid[:,2].reshape(shape_angle[0],shape_angle[1])
            oradius_matrix=oradius_grid[:,2].reshape(shape_angle[0],shape_angle[1])

            iradius_interpolator=RegularGridInterpolator((coords1,coords2),
                                                        iradius_matrix,method='linear',
                                                        bounds_error=False,fill_value=None)
            oradius_interpolator=RegularGridInterpolator((coords1,coords2),
                                                                oradius_matrix,method='linear',
                                                                bounds_error=False,fill_value=None)
            
            iradius_dict=grid_to_dict(iradius_grid)
            oradius_dict=grid_to_dict(oradius_grid)

            guess_inner=[]
            percentual_inner=[]

            for k in range(len(guesses)):

                if not assigned_outer:
                    guess=guesses[k]

                assigned_inner=False

                if k>int(len(guesses)*iteration_limit):
                    possible, best_guess = check_trend2D(guess_inner,percentual_inner,
                                                         guesses[k:],threshold)
                    
                    if possible:
                        guess=best_guess
                        assigned_inner=True
                    else:
                        continue
                
                grid=[]
                coords3=[]
                radius_arr=[]
                coords1_extended=[]
                coords2_extended=[]

                for theta,phi in iradius_dict.keys():

                    inner_radius=iradius_dict[(theta,phi)]
                    outer_radius=oradius_dict[(theta,phi)]

                    # We leave the radius in the innermost loop due to its dependence on
                    # the angular coordinates. We must remember that when passing the 
                    # coordinates to the function

                    thickness=outer_radius-inner_radius
                    rs=np.linspace(inner_radius,outer_radius,
                                   int(guess)+1)
                    
                    # This illustrates the difficulty in writing an optimization algorithm. 
                    # Would the code benefit from the radii having different resolutions?
                    # Possibly yes. This would reduce file size and time needed for interpolation.
                    # But the problem resides in the code itself and running times. In order to do so,
                    # I would need to either go through every permutation of resolution value per theta
                    # phi coordinate, which quickly becomes impracticable, or treat every radius as an 
                    # isolated optimization problem, something that introduces many more calculations 
                    # and is not guaranteed to yield precise results, as the property estimate per position
                    # cannot be assumed to be independent. 
                    
                    # A possible strategy to do so would be to focus on the derivative of the function to 
                    # be gridded instead of the function itself, and by evaluating the best resolution for
                    # a couple representative radii and correlating it to the total slope of the function, 
                    # extrapolating using interpolating the most promising resolution for each coordinate, 
                    # thus skipping the optimization problem altogether whenever possible. Implementing so, 
                    # however, is substantially more complex than it seems and its not guaranteed to yield 
                    # significant improvement or even work, thus making this more of a possible future test
                    # than an upgrade that MUST be done. 

                    coords3.extend(rs)
                    coords1_extended.extend(coords1)
                    coords2_extended.extend(coords2)

                    # The density is coded in such a way that it expects to receive cartesian
                    # coordinates,as this is what hte propagator itself calls it with (a choice
                    # made to keep the propagator separated from any geoemetry modules). 
                    # We must account for that 

                    
                    for r in rs:

                        if func_coord_system=="Cartesian":
                            coord1_temp,coord2_temp,coord3_temp=spherical_to_cartesian(r,theta,phi)
                            if pos_dependent:
                                coord1_temp,coord2_temp,coord3_temp=np.array([coord1_temp,coord2_temp,coord3_temp])+position

                        else:
                            if pos_dependent:
                                coord1_temp,coord2_temp,coord3_temp=spherical_to_cartesian(r,theta,phi)+position
                                coord1_temp,coord2_temp,coord3_temp=cartesian_to_spherical(coord1_temp,coord2_temp,coord3_temp)
                            else:
                                coord1_temp,coord2_temp,coord3_temp=r,theta,phi

                        if pack:
                            grid.append(np.array([r,theta,phi,
                                        function((coord1_temp,coord2_temp,coord3_temp))]))
                        else:
                            grid.append(np.array([r,theta,phi,
                                        function(coord1_temp,coord2_temp,coord3_temp)]))
                            
                grid=np.array(grid)

                grid_test=grid[:,3]
                coords=grid[:,:3]

                # A slower interpolator but it takes ragged arrays as an argument, exactly
                # what we need for the interpolation of our sphere. 
                # I am not too worried about performance here, as the case of a sphere with
                # variable inner and outer surface and variable fast changing density will be
                # genuinely reserved for lower end situations, like simulations using a complete
                # earth model, and these must be ran on a server anyway.
              
                interp = LinearNDInterpolator(coords, 
                                        grid_test,
                                        fill_value=np.nan)
                
                # Ordering is inverted so the input can be provided in a bit 
                # of a more intuitive way.

                # The selection of the sample points has to be done a bit of a different way:
                # First, we use random.choice to select a random set of angular coordinates and
                # interpolate the inner and outer radius in that section of the spherical object,
                # and then we do a random draw of a radius value within the object and interpolate
                # the density, evaluating the precision of the estimate there. We still allow 
                # the separate amount of samples per coordinate, with a standard that has the 
                # same amount of samples per quantity, but some compromises are needed.

                
                interp_coords,func_coords=comparison_coords3DSphere(position,
                                                                    [coords1,coords2],
                                                                    [iradius_interpolator,oradius_interpolator],
                                                                    coords3,
                                                                    samples,sample_fraction,
                                                                    coord_system,func_coord_system,
                                                                    pos_dependent,)
                
                test_points=interp(interp_coords)

                if pack:
                    comparison=function((func_coords[:,0],
                                        func_coords[:,1],
                                        func_coords[:,2]))
                else:
                    comparison=function(func_coords[:,0],
                                        func_coords[:,1],
                                        func_coords[:,2])
                    
                if np.any(comparison==0) or np.any(test_points==0):
                    if np.any(comparison==0):
                        mask=comparison!=0
                    else:
                        mask=test_points!=0
                    comparison=comparison[mask]
                    test_points=test_points[mask]

                mean=np.mean(np.divide(abs(comparison-test_points),comparison))

                if mean<=threshold:
                    return grid
                    
                else:
                    percentual_inner.append(mean)
                    guess_inner.append(guess)     
                    guess_outer.append([guess2,guess])  

                    if assigned_outer or assigned_inner:
                        if assigned_inner:
                            assigned_inner=False
                            continue
                        else:
                            assigned_outer=False
                            break

            percentual.append(percentual_inner)   

    raise RuntimeError("Requested resolution is not achievable. " \
    "Lower threshold and try again") 
   
# As the Block class cannot take height maps for its surfaces (implementing it for a 
# block would be several times more complicated than for the spherical solids), 
# the serialization is several times simpler.               

def serialize3D_block(position,coord_lims,function,guesses=None,
                        samples=None,sample_fraction=0.125,
                        threshold=0.001,iteration_limit=0.5,
                        coord_system="Cartesian",
                        pos_dependent=True,
                        func_coord_system="Cartesian",
                        rotation_history=None,
                        pack=True):
    

    if guesses is not None:
        guess1,guess2,guess3=guesses
        
    else:
        space1=coord_lims[0][1]-coord_lims[0][0]
        space2=coord_lims[1][1]-coord_lims[1][0]
        space3=coord_lims[2][1]-coord_lims[2][0]

        guess1,guess2,guess3=space1/10,space2/10,space3/10

    guesses1 = distance_array(guess1)
    guesses2 = distance_array(guess2)
    guesses3 = distance_array(guess3)

    for guess1 in guesses1:
        
        percentual=[]
        guess_outer=[]
        step_coord=None

        for j in range(len(guesses2)):

            guess2=guesses2[j]
            assigned_outer=False


            if j>int(len(guesses2)*iteration_limit):

                outer_guesses=guesses2[j:]
                new_guesses=[outer_guesses,guesses3]

                possible, best_guesses = check_trend3D(guess_outer,percentual,
                                                       new_guesses,threshold)
                     
                if possible:

                    guess2,guess3 = best_guesses
                    assigned_outer=True

                else:

                    continue

            grid=None

            guess_inner=[]
            percentual_inner=[]
            for k in range(len(guesses3)):

                assigned_inner=False
                if not assigned_outer:
                    guess3=guesses3[k]

                if k>len(guesses3)*iteration_limit:

                    possible,best_guess=check_trend2D(guess_inner,percentual_inner,
                                                      guesses3[k:],threshold)
                    
                    if possible:

                        guess3=best_guess
                        assigned_inner=True

                    else:
                        
                        continue
                        
                if grid is None:
                
                    coords1=np.linspace(coord_lims[0][0],coord_lims[0][1],
                            int(np.ceil((coord_lims[0][1]-coord_lims[0][0])/guess1+1)))
                    
                    if coord_system=="Spherical":
                        coords2=np.arccos(np.linspace(-1,1,int(np.ceil(2/guesses2[0])+1)))

                        coords3=np.linspace(0,np.nextafter(2*np.pi,0),int(np.ceil(2*np.pi/guesses3[1])+1))

                    else:
                        coords2=np.linspace(coord_lims[1][0],coord_lims[1][1],
                            int(np.ceil((coord_lims[1][1]-coord_lims[1][0])/guess2+1)))

                        coords3=np.linspace(coord_lims[2][0],coord_lims[2][1],
                            int(np.ceil((coord_lims[2][1]-coord_lims[2][0])/guess3+1)))

                    shape=[len(coords1),len(coords2),len(coords3),4]

                    actual=coords3[1]-coords3[0]
                    step_coord= actual

                    grid=grid_3D(position,function,
                                [coords1,coords2,coords3],
                                pos_dependent,rotation_history,
                                func_coord_system,pack,coord_system)
                    
                else:

                    grid,coords3,step_coord,shape=refine_3D(position,function,grid,shape,coords3,
                                                            guess3,step_coord,
                                                            rotation_history,pack,
                                                            coord_system,func_coord_system,
                                                            pos_dependent)
                
                interp = RegularGridInterpolator((coords1, coords2,coords3), 
                                        grid[:,3].reshape(shape[0],shape[1],shape[2]),
                                        method="linear",bounds_error=False,fill_value=None)
                
                interp_coords,func_coords=comparison_coords3D(position,
                                                              [coords1,coords2,coords3],
                                                              coord_system,func_coord_system,
                                                              samples,sample_fraction,
                                                              rotation_history,
                                                              pos_dependent)

                test_points=interp(interp_coords)

                if pack:
                    comparison=function((func_coords[:,0],
                                        func_coords[:,1],
                                        func_coords[:,2]))
                else:
                    comparison=function(func_coords[:,0],
                                        func_coords[:,1],
                                        func_coords[:,2])
                    
                # Not really the best solution but the idea behind this solution is that 
                # if you have a point where the density or stopping power is zero, it is 
                # probably a boundary going to vacuum or a discountinuity in a function, 
                # not a persistent feature, and we can safely smooth it out. If someone
                # defines an object with parts of it with zero density... They probably 
                # should think about using the SimulationVolume class instead

                if np.any(comparison==0) or np.any(test_points==0):
                    if np.any(comparison==0):
                        mask=comparison!=0
                    else:
                        mask=test_points!=0
                    comparison=comparison[mask]
                    test_points=test_points[mask]

                # Setting up safeguards for negative reference values is not really necessary
                # because we do not expect density or stopping power to be negative.

                mean=np.mean(np.divide(abs(comparison-test_points),comparison))
                
                if mean<=threshold:
                    return grid
                
                else:
                    percentual_inner.append(mean)
                    guess_inner.append(guess3)     
                    guess_outer.append([guess2,guess3])

                    if assigned_inner or assigned_outer:
                        if assigned_inner:
                            assigned_inner=False
                            continue
                        else:
                            assigned_outer=False
                            break

            percentual.append(percentual_inner) 

    raise RuntimeError("Requested resolution is not achievable. " \
    "Lower threshold and try again")                 
                    

                    
