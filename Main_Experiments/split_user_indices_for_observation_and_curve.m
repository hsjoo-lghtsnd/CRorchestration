function splitIdx = split_user_indices_for_observation_and_curve(userEntry, obsSize)
%SPLIT_USER_INDICES_FOR_OBSERVATION_AND_CURVE
% Split one user's assigned indices into observable and hidden-label parts.

    arguments
        userEntry struct
        obsSize (1,1) {mustBeInteger, mustBePositive}
    end

    N = userEntry.numSamples;

    if obsSize >= N
        error('obsSize must be smaller than userEntry.numSamples.');
    end

    splitIdx = struct();
    splitIdx.userId = userEntry.userId;
    splitIdx.scenarioId = userEntry.scenarioId;
    splitIdx.obsSize = obsSize;
    splitIdx.curveSize = N - obsSize;

    splitIdx.obsGlobalIndices = userEntry.globalIndices(1:obsSize);
    splitIdx.curveGlobalIndices = userEntry.globalIndices(obsSize+1:end);

    splitIdx.obsLocalIndices = userEntry.localIndices(1:obsSize);
    splitIdx.curveLocalIndices = userEntry.localIndices(obsSize+1:end);
end