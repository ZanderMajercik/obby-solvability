local ReplicatedStorage = game:GetService("ReplicatedStorage")
local TweenService = game:GetService("TweenService")
local RunService = game:GetService("RunService")
local TagFunctions = {}

local KILLBLOCK_COLOR = Color3.new(1,0,0)
local CHECKPOINT_COLOR = Color3.new(0,0,1)
local VELOCITY_COLOR = Color3.new(0.5, 0, 0.5)

local TICK = 0.05

-- Does not take "self" as a parameter so that
-- it can be called with TagFunctions["functionName"](part) syntax
function TagFunctions.Damage(part, behavior)

    part.Color = KILLBLOCK_COLOR
    part.Material = Enum.Material.Neon
    part.Transparency = 0.2

	part.Touched:connect(function(hit)
		if hit and hit.Parent and (hit.Parent:FindFirstChild("Humanoid") or hit.Parent:FindFirstChild("NPCHumanoid")) then
			if hit.Parent:FindFirstChild("Humanoid") then
                local pMax = part.Position + part.size * 0.5
                local pMin = part.Position - part.size * 0.5

                local localPos = hit.Parent.HumanoidRootPart.CFrame.Position
                local trueHit = false

                -- Slightly more forgiving collision logic.
                local overMin = pMin.X < localPos.X
                local underMax = pMax.X > localPos.X
                trueHit = trueHit or (overMin and underMax) 

                overMin = pMin.Z < localPos.Z
                underMax = pMax.Z > localPos.Z
                trueHit = trueHit or (overMin and underMax) 
                print("hit", hit, "position", hit.CFrame, "localPos", localPos, "hipHeight", hit.parent.Humanoid.HipHeight)
                
                if trueHit then
                    --if trueHit and localPos.Y <= (CHARACTER_HIP_HEIGHT + 0.0004) then
				    -- hit.Parent.NPCHumanoid.Health = 0
                    hit.Parent.Humanoid.Health = math.max(0, hit.Parent.Humanoid.Health - behavior["damage"])
                end
			else
				-- hit.Parent.NPCHumanoid.Health = 0
                hit.Parent.Humanoid.Health = math.max(0, hit.Parent.Humanoid.Health - behavior["damage"])
			end
		end
    end)
end

local trajectorySuccessEvent = ReplicatedStorage:FindFirstChild("SuccessfulTrajectory")
function TagFunctions.Checkpoint(spawnPart)
    spawnPart.Color = CHECKPOINT_COLOR
    spawnPart.Material = Enum.Material.Neon
    spawnPart.Transparency = 0.2
    --Report a succcessful trajectory and trigger character death.
	-- spawnPart.Touched:connect(function(hit)
    --     -- Touched is an event that is fired when the object is touched
	-- 	local hum = hit.Parent:FindFirstChild("Humanoid")
	-- 	-- If a part in a model touches the part, touch = that part not the model
	-- 	if hum and hum.Health > 0 then
	-- 		-- The thing that touched the checkpoint is alive
	-- 		local player = game:GetService("Players"):GetPlayerFromCharacter(hit.Parent)
	-- 		-- This allows us to find a player with their character (it's very useful)
	-- 		if player then
	-- 			-- Whatever touched the checkpoint is a player
	-- 			local stageInfo = player:WaitForChild("stageInfo")
	-- 			local stage = stageInfo:WaitForChild("Stage")
	-- 			-- Find the stage leader stat
	-- 			if spawnPart.Name:match("Checkpoint " .. tostring(stage.Value + 1)) then
	-- 				-- match() finds certain parts in a string
	-- 				-- So when we do match("%d"), we're finding the numbers in a string
	-- 				-- This is the player's next checkpoint
	-- 				-- This makes it so player's can't skip checkpoints
	-- 				-- or lose progress if they accidently go backwards
	-- 				stage.Value = stage.Value + 1
	-- 			end
	-- 		end
	-- 	end
    -- end)
end

function TagFunctions.Velocity(part, behavior)
    part.Color = VELOCITY_COLOR
    local velocity = Vector3.new(behavior["velocity"][1], behavior["velocity"][3], behavior["velocity"][2])
    part.AssemblyLinearVelocity = velocity
end

function TagFunctions.Disappear(part, behavior)
    local db = Instance.new("BoolValue", part)
    db.Name = "db"
    db.Value = false

    part.Touched:connect(function(hit)
        local db = part:WaitForChild("db")
        if db.Value == false then
            db.Value = true
            for i = 1, behavior["disappear_delay"] do
                part.Transparency = i / behavior["disappear_delay"]
                wait(TICK)
            end
            part.CanCollide = false
            wait(behavior["reappear_delay"] * TICK)
            part.CanCollide = true
            part.Transparency = 0
            db.Value = false
        end
    end)
end

function TagFunctions.Motion(part, behavior)
    local alignPosition = Instance.new("AlignPosition", part)
    local alignOrientation = Instance.new("AlignOrientation", part)
    local easingStyle = Enum.EasingStyle[behavior["easing_style"]]
    local easingDirection = Enum.EasingDirection[behavior["easing_direction"]]
    local startPosition = Vector3.new(
        behavior["start_position"][1],
        behavior["start_position"][3],
        behavior["start_position"][2]
    )
    local endPosition = Vector3.new(
        behavior["end_position"][1],
        behavior["end_position"][3],
        behavior["end_position"][2]
    )
    print(startPosition, endPosition)
    
    local goal = {}
    goal.Position = endPosition

    -- animate the part
    if behavior["type"] == "BehaviorType.Persistent" then
        local tweenInfo = TweenInfo.new(behavior["time"], easingStyle, easingDirection, -1, behavior["reverses"])
        local tween = TweenService:Create(part, tweenInfo, goal)
        print("PersistentMotion")
        tween:Play()
    elseif behavior["type"] == "BehaviorType.Touch" then
        local tweenInfo = TweenInfo.new(behavior["time"], easingStyle, easingDirection, 0, false)
        local tween = TweenService:Create(part, tweenInfo, goal)
        part.Touched:connect(function(hit)
            print("TouchMotion")
            tween:Play()
        end)
    end
end

function TagFunctions.Rotate(part, behavior)
    RunService.Stepped:connect(function(_, deltaTime)
        part.CFrame = part.CFrame * CFrame.Angles(0, 0, behavior["angular"] * math.pi / 90)
    end)
end

function TagFunctions.Carry(part)
    -- apply velocity to the part so that the character moves with it when on top
    local lastPosition = part.Position
    RunService.Stepped:connect(function (_, deltaTime)
        local currPosition = part.Position
        local deltaPosition = currPosition - lastPosition
        local velocity = deltaPosition / deltaTime
        part.AssemblyLinearVelocity = velocity
        lastPosition = currPosition
    end)
end

return TagFunctions