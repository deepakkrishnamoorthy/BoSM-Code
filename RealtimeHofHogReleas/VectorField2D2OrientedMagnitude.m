% orientedMagnitude = VectorField2D2OrientedMagnitude(numOrientations, V1, V2)
%
% Make oriented magnitude for a 2d vector field. Perform linear interpolation
% between between orientations
%
% numOrientations:      The number of orientations.
% V1:                   N x M matrix containing direction in X1 OR
%                       N x M matrix of complex number containing full
%                       vector field. In this case V2 is ignored
% V2:                   N x M matrix containing direction in X2
%
% orientedMagnitude:    N x M x numOrientations matrix of oriented magnitudes
%
%    Jasper Uijlings - 2013
