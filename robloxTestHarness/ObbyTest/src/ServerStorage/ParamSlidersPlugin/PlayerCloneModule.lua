-- PlayerCloneModule.lua
-- Create this as a ModuleScript in your plugins folder or in ReplicatedStorage
local PlayerCloneModule = {}

-- Variables
local clones = {} -- Table to store references to all created clones
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local CollectionService = game:GetService("CollectionService")


-- Function to get the current player in Studio
function PlayerCloneModule.getCurrentPlayer()
    local player = nil
    
    -- First try getting the local player directly in Studio
    if RunService:IsStudio() then
        player = Players.LocalPlayer
    end
    
    -- If that fails, get the first player in the game
    if not player and #Players:GetPlayers() > 0 then
        player = Players:GetPlayers()[1]
    end
    
    return player
end

-- Create a clone of a player character
function PlayerCloneModule.createClone(options)
    options = options or {}

    -- Get player
    local player = options.player or PlayerCloneModule.getCurrentPlayer()
    if not player or not player.Character then
        warn("PlayerCloneModule: Player or character not found")
        return nil
    end
    
    -- Setup container folder if specified or needed
    local parent = options.parent
    if not parent then
        parent = workspace:FindFirstChild("PlayerClones")
        if not parent then
            parent = Instance.new("Folder")
            parent.Name = "PlayerClones"
            parent.Parent = workspace
        end
    end
    
    -- Create clone
    
    player.Character.Archivable = true
    local clone = player.Character:Clone()
    player.Character.Archivable = false
    clone.Name = options.name or (player.Name .. "_Clone")
    clone.Parent = parent

    -- Set massless to make jump height consistent.
    -- Otherwise, jump height is dependent on inertia.
    for i, child in clone:GetChildren() do
        if child:IsA("MeshPart") then
            child.Massless = true
        end
    end
    
    -- Setup the humanoid
    local humanoid = clone:FindFirstChildOfClass("Humanoid")
    humanoid.DisplayName = ""

    humanoid.WalkSpeed = 16
    humanoid.JumpHeight = 7.159
    --modelHumanoid.JumpPower = 31.005
    humanoid.UseJumpPower = false

    -- Position the clone
    local rootPart = clone:WaitForChild("HumanoidRootPart")
        -- Transfer ownership to the server
    rootPart:SetNetworkOwner(nil)
    if rootPart then
        -- Position based on provided options or default to current player position
        if options.position then
            if typeof(options.position) == "Vector3" then
                rootPart.CFrame = CFrame.new(options.position)
            elseif typeof(options.position) == "CFrame" then
                rootPart.CFrame = options.position
            end
        elseif options.offset and player.Character:FindFirstChild("HumanoidRootPart") then
            -- Use offset from player if provided
            local offset = options.offset
            if typeof(offset) == "Vector3" then
                rootPart.CFrame = player.Character.HumanoidRootPart.CFrame * CFrame.new(offset)
            elseif typeof(offset) == "CFrame" then
                rootPart.CFrame = player.Character.HumanoidRootPart.CFrame * offset
            end
        elseif player.Character:FindFirstChild("HumanoidRootPart") then
            -- Default to 3 studs to the right
            rootPart.CFrame = player.Character.HumanoidRootPart.CFrame * CFrame.new(3, 0, 0)
        end
        
        -- Anchor the root part
        rootPart.Anchored = options.anchored ~= false -- Default to true
    end
    
    -- Add a highlight if requested
    if options.highlight then
        local highlight = Instance.new("Highlight")
        highlight.FillColor = options.highlightColor or Color3.fromRGB(0, 200, 255)
        highlight.OutlineColor = options.outlineColor or Color3.fromRGB(255, 255, 255)
        highlight.FillTransparency = options.fillTransparency or 0.5
        highlight.OutlineTransparency = options.outlineTransparency or 0
        highlight.Parent = clone
    end
    
    -- Parent the clone
    clone.Parent = parent
    
    -- Store reference to the clone
    table.insert(clones, clone)

    CollectionService:AddTag(clone, "NoDamage")
    
    return clone
end

-- Hide character
function PlayerCloneModule.hideClone(clone)
    for i,v in pairs(clone:GetDescendants()) do -- loop through everything in the character
        if v:IsA("BasePart") or v:IsA("Decal") then -- if it is a part
            v.Transparency = 1 -- make it invisible
        end
    end
end

-- Show character
function PlayerCloneModule.showClone(clone)
    if clone:FindFirstChildOfClass("Highlight") then
        clone.Highlight.FillTransparency = 0.5
    end
    for i,v in pairs(clone:GetDescendants()) do -- loop through everything in the character
        if (v:IsA("BasePart") or v:IsA("Decal")) and not string.find(v.Name, "HumanoidRootPart") then -- if it is a part
            v.Transparency = 0 -- make it invisible
        end
    end
end

-- Kill a specific clone
function PlayerCloneModule.killClone(clone)
    if clone:FindFirstChildOfClass("Highlight") then
        clone.Highlight.FillTransparency = 0.0
    end
    if not clone then
        warn("PlayerCloneModule: No clone specified to kill")
        return false
    end
    
    local humanoid = clone:FindFirstChildOfClass("Humanoid")
    if humanoid then
        -- humanoid.Health = 0
        return true
    else
        warn("PlayerCloneModule: No humanoid found in clone")
        return false
    end
end



-- Reposition a clone
-- Note that the "touched" event does not activate in response to 
-- this repositioning because it is a non-physics based movement.
function PlayerCloneModule.moveClone(clone, newPosition, keepOrientation)
    if clone:FindFirstChildOfClass("Highlight") then
        clone.Highlight.FillTransparency = 0.75
    end
    if not clone then
        warn("PlayerCloneModule: No clone specified to move")
        return false
    end
    
    local rootPart = clone:WaitForChild("HumanoidRootPart")
    if not rootPart then
        warn("PlayerCloneModule: No HumanoidRootPart found in clone")
        return false
    end

    -- Disable collision.
    for i, child in clone:GetChildren() do
        if child:IsA("BasePart") then
            child.CanCollide = false
        end
    end
    
    -- Handle different position types
    if typeof(newPosition) == "Vector3" then
        if keepOrientation then
            -- Keep the same orientation, only change position
            local currentCFrame = rootPart.CFrame
            rootPart.CFrame = CFrame.new(newPosition) * CFrame.Angles(
                currentCFrame:ToOrientation()
            )
        else
            rootPart.CFrame = CFrame.new(newPosition)
        end
    elseif typeof(newPosition) == "CFrame" then
        rootPart.CFrame = newPosition
    else
        warn("PlayerCloneModule: Invalid position type. Expected Vector3 or CFrame")
        return false
    end

    -- Make sure root part faces forward
    rootPart.CFrame = rootPart.CFrame * CFrame.Angles(0, math.pi, 0)
    
    return true
end

local humanoidMoveTestVec = {}
humanoidMoveTestVec[1] = Vector3.new(0, 0, 1)
humanoidMoveTestVec[2] = Vector3.new(-1, 0, 1)
humanoidMoveTestVec[3] = Vector3.new(-1, 0, 0)
humanoidMoveTestVec[4] = Vector3.new(-1, 0, -1)
humanoidMoveTestVec[5] = Vector3.new(0, 0, -1)
humanoidMoveTestVec[6] = Vector3.new(1, 0, -1)
humanoidMoveTestVec[7] = Vector3.new(1, 0, 0)
humanoidMoveTestVec[8] = Vector3.new(1, 0, 1)

function PlayerCloneModule.cloneAction(clone, action)
    if clone:FindFirstChildOfClass("Highlight") then
        clone.Highlight.FillTransparency = 0.75
    end
    if not clone then
        warn("PlayerCloneModule: No clone specified to move")
        return false
    end

    if not clone:FindFirstChild("HumanoidRootPart") then
        warn("No root part.")
        return
    end

    -- Convert from action direction indices to movement vectors
    local moveVector = humanoidMoveTestVec[action[2] + 1]

    if action[1] ~= 1 then
        moveVector = Vector3.new(0,0,0)
    end
 
    -- Take the final action
    local cloneHumanoid = clone:WaitForChild("Humanoid")
    cloneHumanoid:Move(moveVector)
    cloneHumanoid.Jump = (action[4] == 1)
    return true
end

-- Get all created clones
function PlayerCloneModule.getClones()
    -- Clean up any destroyed clones from our tracking table
    for i = #clones, 1, -1 do
        if not clones[i] or not clones[i].Parent then
            table.remove(clones, i)
        end
    end
    
    return clones
end

-- Destroy a specific clone
function PlayerCloneModule.destroyClone(clone)
    if not clone then return false end
    
    -- Remove from tracking table
    for i, trackedClone in ipairs(clones) do
        if trackedClone == clone then
            table.remove(clones, i)
            break
        end
    end
    
    -- Destroy the clone
    clone:Destroy()
    return true
end

-- Destroy all created clones
function PlayerCloneModule.destroyAllClones()
    for _, clone in ipairs(clones) do
        if clone and clone.Parent then
            clone:Destroy()
        end
    end
    
    clones = {}
    return true
end

return PlayerCloneModule