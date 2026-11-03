local ReplicatedStorage = game:GetService("ReplicatedStorage")
local TweenService = game:GetService("TweenService")
local RunService = game:GetService("RunService")
local CollectionService = game:GetService("CollectionService")
local TrainingBehaviorFunctions = {}


--local trajectorySuccessEvent = ReplicatedStorage:FindFirstChild("SuccessfulTrajectory")
function TrainingBehaviorFunctions.KillAndReportEpisode(part, episode)
    --Report a succcessful trajectory and trigger character death.
	killAndReportEpisodeConnection = part.Touched:connect(function(hit)
        -- Touched is an event that is fired when the object is touched
        if hit and hit.Parent and hit.Parent:FindFirstChild("Humanoid") and not CollectionService:HasTag(hit.Parent, "NoDamage") then
	    	-- Report the successful trajectory
	    	-- trajectorySuccessEvent:Fire()
			episode.success = true
			wait(0.25)
	    	hit.Parent.Humanoid.Health = 0
	    end
    end)
	return killAndReportEpisodeConnection
end

return TrainingBehaviorFunctions